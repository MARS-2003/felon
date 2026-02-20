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
        # Get all subdirectories
        folders = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
        folders.sort()

        print(f"--- Found {len(folders)} folders to process ---")

        for folder_path in folders:
            folder_name = os.path.basename(folder_path)
            print(f"\n>>> Processing: {folder_name}")

            # Create processes for the current folder
            p_stress = multiprocessing.Process(target=stress.run, args=(folder_path,))
            p_logger = multiprocessing.Process(target=logger.run, args=(folder_path,))

            print(f"--- Starting Concurrent Run for {folder_name} ---")
            p_logger.start()
            p_stress.start()

            # Wait for stress to finish
            p_stress.join()

            # Shutdown logger
            if p_logger.is_alive():
                p_logger.terminate()
                p_logger.join()

            print(f"--- Finished processing {folder_name} ---")

        print("\n--- All folders complete ---")
