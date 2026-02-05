#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import csv

# === Path to CSV file ===
csv_path = input("Enter path to CSV file: ").strip()

# === Data containers ===
times = []
powers = []
rams = []
temps = []
gpu_usages = []

# === Read CSV ===
with open(csv_path, newline='') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        times.append(float(row["Time_s"]))
        powers.append(float(row["Power_W"]))
        rams.append(float(row["RAM_MB"]))
        temps.append(float(row["CPU_Temp_C"]))
        gpu_usages.append(float(row["GPU_Usage_percent"]))

# === Plot with independent XY axes ===
fig, axs = plt.subplots(4, 1, figsize=(12, 14))

axs[0].plot(times, powers, color="red")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Power (W)")
axs[0].grid(True)
axs[0].set_title("Power")

axs[1].plot(times, rams, color="blue")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("RAM Used (MB)")
axs[1].grid(True)
axs[1].set_title("RAM Usage")

axs[2].plot(times, temps, color="green")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("CPU Temp (°C)")
axs[2].grid(True)
axs[2].set_title("CPU Temperature")

axs[3].plot(times, gpu_usages, color="purple")
axs[3].set_xlabel("Time (s)")
axs[3].set_ylabel("GPU Usage (%)")
axs[3].grid(True)
axs[3].set_title("GPU Usage")

plt.tight_layout()

# === Save the figure in same directory as CSV ===
save_dir = os.path.dirname(csv_path)
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
png_path = os.path.join(save_dir, f"logger_plot_{timestamp}.png")
plt.savefig(png_path)
plt.show()
plt.close()

print(f"Plot saved to: {png_path}")
