#use this script to generate an automated report by comparing all the faulty _stress.csv and _logger.csv. 
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def get_class_counts(class_id_str):
    if pd.isna(class_id_str) or str(class_id_str).strip() == "":
        return Counter()
    return Counter(map(int, str(class_id_str).split()))

def analyze_pair(base_stress, test_stress, base_log, test_log):
    df_bs, df_ts = pd.read_csv(base_stress), pd.read_csv(test_stress)
    df_bl, df_tl = pd.read_csv(base_log), pd.read_csv(test_log)
    
    # --- Detection Analysis ---
    common_frames = sorted(list(set(df_bs['Frame_Index']) & set(df_ts['Frame_Index'])))
    t_base_dets, t_test_dets, t_retention, t_drops, t_hallucinations = 0, 0, 0, 0, 0
    for idx in common_frames:
        counts_b = get_class_counts(df_bs[df_bs['Frame_Index'] == idx].iloc[0]['Class_IDs'])
        counts_t = get_class_counts(df_ts[df_ts['Frame_Index'] == idx].iloc[0]['Class_IDs'])
        for cls in (set(counts_b.keys()) | set(counts_t.keys())):
            b_cnt, t_cnt = counts_b[cls], counts_t[cls]
            t_base_dets += b_cnt
            t_test_dets += t_cnt
            t_retention += min(b_cnt, t_cnt)
            t_drops += max(0, b_cnt - t_cnt)
            t_hallucinations += max(0, t_cnt - b_cnt)

    base_val = max(1, t_base_dets)
    
    # --- Resource and Comparison Analysis ---
    b_pwr, b_ram, b_temp, b_fps = df_bl['Power_W'].mean(), df_bl['RAM_MB'].max(), df_bl['CPU_Temp_C'].mean(), df_bs['FPS_Instant'].mean()
    t_pwr, t_ram, t_temp, t_fps = df_tl['Power_W'].mean(), df_tl['RAM_MB'].max(), df_tl['CPU_Temp_C'].mean(), df_ts['FPS_Instant'].mean()

    return {
        "Test_Case": "", 
        "Retention_%": round((t_retention / base_val) * 100, 2),
        "Drop_%": round((t_drops / base_val) * 100, 2),
        "Hallucin_%": round((t_hallucinations / base_val) * 100, 2),
        "FPS_Avg": round(t_fps, 2),
        "FPS_Delta": round(t_fps - b_fps, 2),
        "Power_W": round(t_pwr, 2),
        "Power_Delta": round(t_pwr - b_pwr, 2),
        "RAM_Peak": round(t_ram, 1),
        "RAM_Delta": round(t_ram - b_ram, 1),
        "Temp_Avg": round(t_temp, 1),
        "Temp_Delta": round(t_temp - b_temp, 1),
        "GPU_%": round(df_tl['GPU_Usage_percent'].mean(), 1)
    }

def format_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=16) # Removed 'pad' to fix AttributeError
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment('right')

def generate_charts(df, save_dir):
    plt.style.use('ggplot')

    # Chart 1: Inference and Performance
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
    df.plot(x='Test_Case', y=['Retention_%', 'Drop_%', 'Hallucin_%'], kind='bar', ax=ax1, 
            color=['#2E7D32', '#C62828', '#EF6C00'], width=0.8)
    format_ax(ax1, "Detection Robustness", "Percentage (%)")

    fps_colors = ['#C62828' if x < 0 else '#2E7D32' for x in df['FPS_Delta']]
    ax2.bar(df['Test_Case'], df['FPS_Delta'], color=fps_colors)
    ax2.axhline(0, color='black', linewidth=1)
    format_ax(ax2, "FPS Change relative to Baseline", "Delta FPS")
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4) # Manual spacing to replace 'pad'
    fig1.savefig(os.path.join(save_dir, "inference_performance.png"))

    # Chart 2: Hardware Resources
    fig2, axes = plt.subplots(2, 2, figsize=(24, 18))
    
    pwr_colors = ['#C62828' if x > 0 else '#2E7D32' for x in df['Power_Delta']]
    axes[0,0].bar(df['Test_Case'], df['Power_Delta'], color=pwr_colors)
    format_ax(axes[0,0], "Power Change (Delta)", "Watts (W)")

    axes[0,1].bar(df['Test_Case'], df['RAM_Delta'], color='#5D4037')
    format_ax(axes[0,1], "RAM Change (Delta)", "MB")

    tmp_colors = ['#D84315' if x > 0 else '#0277BD' for x in df['Temp_Delta']]
    axes[1,0].bar(df['Test_Case'], df['Temp_Delta'], color=tmp_colors)
    format_ax(axes[1,0], "Temperature Change (Delta)", "Celsius (C)")

    axes[1,1].bar(df['Test_Case'], df['GPU_%'], color='#455A64')
    format_ax(axes[1,1], "Average GPU Usage", "Usage (%)")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    fig2.savefig(os.path.join(save_dir, "hardware_resources.png"))

def main():
    base_stress_path = input("Path to Baseline stress.csv: ").strip()
    base_logger_path = base_stress_path.replace("_stress.csv", "_logger.csv")
    folder_path = input("Path to folder containing test pairs: ").strip()

    results = []
    for f in os.listdir(folder_path):
        if f.endswith("_stress.csv") and os.path.abspath(os.path.join(folder_path, f)) != os.path.abspath(base_stress_path):
            ts, tl = os.path.join(folder_path, f), os.path.join(folder_path, f.replace("_stress.csv", "_logger.csv"))
            if os.path.exists(tl):
                metrics = analyze_pair(base_stress_path, ts, base_logger_path, tl)
                metrics['Test_Case'] = f.replace("_stress.csv", "")
                results.append(metrics)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(folder_path, "analysis_results.csv"), index=False)
        with open(os.path.join(folder_path, "analysis_report.txt"), "w") as f:
            f.write("-" * 150 + "\nROBUSTNESS AND RESOURCE COMPARISON REPORT\n" + "-" * 150 + "\n")
            f.write(f"Baseline file: {os.path.basename(base_stress_path)}\n\n")
            
            # Metric Definitions
            f.write("METRIC DEFINITIONS:\n")
            f.write("Retention %:   Percentage of original objects correctly identified during stress.\n")
            f.write("Drop %:        Percentage of objects missed during stress compared to baseline.\n")
            f.write("Hallucin %:    Percentage of extra objects detected that were not in the baseline.\n")
            f.write("FPS_Delta:     The loss or gain in frames per second compared to the baseline.\n")
            f.write("Power_Delta:   The increase or decrease in power draw (Watts) compared to the baseline.\n")
            f.write("RAM_Delta:     The increase in peak memory usage compared to the baseline.\n")
            f.write("Temp_Delta:    The increase in average CPU temperature compared to the baseline.\n")
            f.write("-" * 150 + "\n\n")
            
            f.write(df.to_string(index=False))
            f.write("\n\n" + "-" * 150 + "\n")
            f.write("Delta values show the difference between the test case and the baseline.\n")
            f.write("-" * 150 + "\n")
        
        generate_charts(df, folder_path)
        print(f"\nCompleted. Files saved in: {folder_path}")
    else:
        print("No matching files found.")

if __name__ == "__main__":
    main()
