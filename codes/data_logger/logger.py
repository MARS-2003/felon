#!/usr/bin/env python3
import os
import time
import csv
from datetime import datetime
from jtop import jtop
import math

def run(folder_path):
    # === Match directory of the Folder ===
    save_dir = os.path.dirname(folder_path)
    folder_basename = os.path.basename(folder_path)
    csv_path = os.path.join(save_dir, f"{folder_basename}_logger.csv")

    print(f"Starting logger. Saving to: {csv_path}")

    with jtop() as jetson:
        start_time = time.time()
        try:
            with open(csv_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Time_s", "Power_W", "RAM_MB", "CPU_Temp_C", "GPU_Usage_percent"])

                while jetson.ok():
                    current_time = time.time() - start_time

                    # --- Power Extraction ---
                    p_tot = jetson.power.get('tot', {})
                    power = p_tot.get('avg', 0) / 1000.0 if isinstance(p_tot, dict) else 0.0

                    # --- RAM Extraction ---
                    ram_data = jetson.memory.get('RAM', {})
                    ram_used = ram_data.get('used', 0) / 1024.0

                    # --- Temp Extraction ---
                    temp_data = jetson.temperature.get('CPU', {})
                    cpu_temp = temp_data.get('temp', math.nan)

                    # --- GPU Extraction (Robust check) ---
                    gpu_data = jetson.gpu.get('gpu', {})
                    gpu_usage = gpu_data.get('status', {}).get('load', 0.0) 
                    if gpu_usage == 0.0:
                        gpu_usage = gpu_data.get('load', 0.0)

                    # Write to CSV
                    writer.writerow([
                        f"{current_time:.3f}", 
                        f"{power:.3f}", 
                        f"{ram_used:.1f}", 
                        f"{cpu_temp:.1f}", 
                        f"{gpu_usage:.1f}"
                    ])
                    
                    csv_file.flush()
                    time.sleep(0.1)  # 10 Hz

        except Exception:
            pass

    print(f"Logger CSV saved to: {csv_path}")
