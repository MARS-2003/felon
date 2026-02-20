#!/usr/bin/env python3
import os
import time
import csv
from jtop import jtop
import math

def run(folder_path, results_dir):
    folder_basename = os.path.basename(folder_path)
    csv_path = os.path.join(results_dir, f"{folder_basename}_logger.csv")

    with jtop() as jetson:
        start_time = time.time()
        try:
            with open(csv_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Time_s", "Power_W", "RAM_MB", "CPU_Temp_C", "GPU_Usage_percent"])

                while jetson.ok():
                    current_time = time.time() - start_time
                    
                    p_tot = jetson.power.get('tot', {})
                    power = p_tot.get('avg', 0) / 1000.0 if isinstance(p_tot, dict) else 0.0
                    
                    ram_data = jetson.memory.get('RAM', {})
                    ram_used = ram_data.get('used', 0) / 1024.0
                    
                    temp_data = jetson.temperature.get('CPU', {})
                    cpu_temp = temp_data.get('temp', math.nan)
                    
                    gpu_data = jetson.gpu.get('gpu', {})
                    gpu_usage = gpu_data.get('status', {}).get('load', 0.0) 
                    if gpu_usage == 0.0:
                        gpu_usage = gpu_data.get('load', 0.0)

                    writer.writerow([
                        f"{current_time:.3f}", 
                        f"{power:.3f}", 
                        f"{ram_used:.1f}", 
                        f"{cpu_temp:.1f}", 
                        f"{gpu_usage:.1f}"
                    ])
                    
                    csv_file.flush()
                    time.sleep(0.1)

        except Exception:
            pass
