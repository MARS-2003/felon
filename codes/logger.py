#!/usr/bin/env python3
import os
import time
import csv
from datetime import datetime
from jtop import jtop  # Updated import
import math

# === Prepare save directory ===
save_dir = os.path.expanduser("~/Downloads/original")
os.makedirs(save_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(save_dir, f"logger_data_{timestamp}.csv")

# === Start jtop ===
print("Starting logger... Press Ctrl+C to stop and save CSV.")

with jtop() as jetson:
    start_time = time.time()
    try:
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Time_s", "Power_W", "RAM_MB", "CPU_Temp_C", "GPU_Usage_percent"])

            while jetson.ok():
                current_time = time.time() - start_time

                # --- Power Extraction ---
                # jtop returns power in mW usually, converting to W
                p_tot = jetson.power.get('tot', {})
                power = p_tot.get('avg', 0) / 1000.0 if isinstance(p_tot, dict) else 0.0

                # --- RAM Extraction (KB to MB) ---
                ram_data = jetson.memory.get('RAM', {})
                ram_used = ram_data.get('used', 0) / 1024.0

                # --- Temp Extraction ---
                temp_data = jetson.temperature.get('CPU', {})
                cpu_temp = temp_data.get('temp', math.nan)

                # --- GPU Extraction ---
                gpu_data = jetson.gpu.get('gpu', {})
                gpu_status = gpu_data.get('status', {})
                gpu_usage = gpu_status.get('load', 0.0)

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

    except KeyboardInterrupt:
        print("\nStopping logger and closing CSV.")

print(f"CSV saved to: {csv_path}")
