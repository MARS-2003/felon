#this is the main control script, save this script along with the other scripts in this directory in the save directory on your jetson, and just run python3 combined.py
import multiprocessing
import os
import stress
import logger

if __name__ == '__main__':
    # CRITICAL: Fixes CUDA context issues in subprocesses
    multiprocessing.set_start_method('spawn', force=True)

    parent_dir = input("Enter full path to the directory containing your folders: ").strip()
    if not os.path.isdir(parent_dir):
        print("Error: Directory not found.")
    else:
        # Create a single central results directory
        results_dir = os.path.join(parent_dir, "benchmark_results")
        os.makedirs(results_dir, exist_ok=True)

        # Get all subdirectories, excluding the results folder itself
        folders = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
                   if os.path.isdir(os.path.join(parent_dir, d)) and d != "benchmark_results"]
        folders.sort()

        total_folders = len(folders)
        print(f"--- Found {total_folders} folders to process ---")
        print(f"--- Results will be saved to: {results_dir} ---")

        for index, folder_path in enumerate(folders, 1):
            folder_name = os.path.basename(folder_path)
            print(f"\n[{index}/{total_folders}] >>> Processing: {folder_name}")

            # Create processes
            p_stress = multiprocessing.Process(target=stress.run, args=(folder_path, results_dir))
            p_logger = multiprocessing.Process(target=logger.run, args=(folder_path, results_dir))

            p_logger.start()
            p_stress.start()

            # Wait for inference to finish
            p_stress.join()

            # Shutdown logger
            if p_logger.is_alive():
                p_logger.terminate()
                p_logger.join()

        print("\n--- All processes complete. Check the 'benchmark_results' folder. ---")
