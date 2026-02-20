#this one doesn't save the annotated pictures. Use this one if you need accurate results on resource logging
import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import csv
import atexit
import shutil

def run(folder_path):
    # 1. Initialize inside the function for Process safety
    ENGINE_PATH = os.path.expanduser("~/Downloads/onnx/my_model.engine")
    CLASSES_TXT = os.path.expanduser("~/Downloads/onnx/classes.txt")
    BATCH, CLASS_CONF_THRES, OBJ_THRES = 1, 0.55, 0.25
    NMS_IOU, MIN_BOX_AREA, TOP_K, PRE_N = 0.45, 16, 60, 200

    PARENT_DIR = os.path.dirname(folder_path)
    folder_basename = os.path.basename(folder_path)
    csv_filename = os.path.join(PARENT_DIR, f"{folder_basename}_stress.csv")
    
    # Open with buffering=1 (line buffered) to prevent empty files on crash
    csv_file = open(csv_filename, 'w', newline='', buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame_Index", "Latency_Sec", "FPS_Instant", "Detection_Count", "Class_IDs"])

    # --- CUDA / TensorRT Setup ---
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    input_idx = 0 if engine.binding_is_input(0) else 1
    output_idx = 1 - input_idx
    in_shape = tuple(int(x) for x in engine.get_binding_shape(input_idx))
    out_shape = tuple(int(x) for x in engine.get_binding_shape(output_idx))
    _, C, H_model, W_model = in_shape

    stream = cuda.Stream()
    bindings, host_buffers, dev_buffers = [None]*2, {}, {}
    for i in range(engine.num_bindings):
        shape = tuple(int(x) if int(x)>0 else 1 for x in engine.get_binding_shape(i))
        dtype_np = trt.nptype(engine.get_binding_dtype(i))
        host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype_np)
        dev_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings[i], host_buffers[i], dev_buffers[i] = int(dev_mem), (host_mem, shape), dev_mem

    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
    def xywh2xyxy(xywh):
        xc, yc, w, h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
        return np.stack([xc - 0.5 * w, yc - 0.5 * h, xc + 0.5 * w, yc + 0.5 * h], axis=1)

    def nms_numpy(boxes, scores, iou_thres):
        if boxes.shape[0] == 0: return np.array([], dtype=np.int32)
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1: break
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
            iou = (w * h) / (areas[i] + areas[order[1:]] - (w * h) + 1e-9)
            order = order[np.where(iou <= iou_thres)[0] + 1]
        return np.array(keep, dtype=np.int32)

    _frame_idx = 0
    image_files = sorted([os.path.join(r, f) for r, _, fs in os.walk(folder_path) for f in fs if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    try:
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None: continue
            
            t_start = time.time()
            
            img_resized = cv2.resize(img, (W_model, H_model))
            arr = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = np.ascontiguousarray(arr.transpose(2,0,1).reshape(BATCH, C, H_model, W_model))
            
            # Inference
            in_host, _ = host_buffers[input_idx]
            np.copyto(np.frombuffer(in_host, dtype=inp.dtype, count=inp.size), inp.ravel())
            cuda.memcpy_htod_async(dev_buffers[input_idx], in_host, stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            out_host, out_shape_buf = host_buffers[output_idx]
            cuda.memcpy_dtoh_async(out_host, dev_buffers[output_idx], stream)
            stream.synchronize()

            # Post-processing
            preds = np.array(out_host, copy=False).reshape(out_shape_buf).transpose(0,2,1)[0]
            xywh, obj_raw, class_raw = preds[:,:4], preds[:,4], preds[:,5:]
            obj_sig, cls_sig = sigmoid(obj_raw), sigmoid(class_raw)
            cls_max, cls_argmax = cls_sig.max(axis=1), np.argmax(cls_sig, axis=1)
            rank_scores = obj_sig * cls_max
            k = min(PRE_N, rank_scores.shape[0])
            cand_idx = np.argpartition(-rank_scores, k-1)[:k] if k < rank_scores.shape[0] else np.arange(rank_scores.shape[0])
            
            cand_boxes = xywh2xyxy(xywh[cand_idx])
            cand_scores, cand_cls, cand_obj = cls_max[cand_idx], cls_argmax[cand_idx], obj_sig[cand_idx]
            mask = (cand_scores >= CLASS_CONF_THRES) & (cand_obj >= OBJ_THRES)
            cand_boxes, cand_scores, cand_cls = cand_boxes[mask], cand_scores[mask], cand_cls[mask]
            
            areas = (cand_boxes[:,2]-cand_boxes[:,0])*(cand_boxes[:,3]-cand_boxes[:,1])
            keep_area = np.where(areas >= MIN_BOX_AREA)[0]
            cand_boxes, cand_scores, cand_cls = cand_boxes[keep_area], cand_scores[keep_area], cand_cls[keep_area]
            
            if cand_boxes.shape[0] > TOP_K:
                idx_topk = np.argsort(-cand_scores)[:TOP_K]
                cand_boxes, cand_scores, cand_cls = cand_boxes[idx_topk], cand_scores[idx_topk], cand_cls[idx_topk]

            keep = nms_numpy(cand_boxes, cand_scores, NMS_IOU)
            final_cls = cand_cls[keep]
            
            latency = time.time() - t_start
            csv_writer.writerow([_frame_idx, f"{latency:.6f}", f"{1.0/latency:.2f}" if latency>0 else 0, len(final_cls), " ".join(str(int(c)) for c in final_cls.tolist())])
            _frame_idx += 1
    finally:
        csv_file.close()
        print(f"Stress log saved to: {csv_filename}")
