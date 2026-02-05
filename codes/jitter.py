import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_path = input("Enter the path to the latency CSV file: ").strip()

    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' does not exist.")
        return

    try:
        df = pd.read_csv(csv_path)
        
        df['Latency_ms'] = df['Latency_Sec'] * 1000
        df['Jitter_ms'] = df['Latency_ms'].diff().abs()

        file_dir = os.path.dirname(csv_path)
        file_name = os.path.splitext(os.path.basename(csv_path))[0]
        save_path = os.path.join(file_dir, f"{file_name}_jitter_analysis.png")

        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.plot(df['Frame_Index'], df['Latency_ms'], label='Latency', color='#1f77b4')
        plt.title('Frame Latency ($ms$)', fontsize=14)
        plt.ylabel('Time ($ms$)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(df['Frame_Index'], df['Jitter_ms'], label='Jitter', color='#d62728')
        plt.title('Jitter ($ms$)', fontsize=14)
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Variation ($ms$)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        
        print(f"\nResults saved to: {save_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
