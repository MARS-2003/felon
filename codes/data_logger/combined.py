#this is the main control script, save this script along with the other scripts in this directory in the save directory on your jetson, and just run python3 combined.py
import multiprocessing
import os
import stress
import logger

if __name__ == '__main__':
    # CRITICAL: Fixes CUDA context issues in subprocesses
    multiprocessing.set_start_method('spawn', force=True)

    zip_path = input("Enter full path to ZIP file: ").strip()
    if not os.path.isfile(zip_path):
        print("Error: File not found.")
    else:
        # Create processes
        p_stress = multiprocessing.Process(target=stress.run, args=(zip_path,))
        p_logger = multiprocessing.Process(target=logger.run, args=(zip_path,))

        print("--- Starting Concurrent Run ---")
        p_logger.start()
        p_stress.start()

        # Wait for stress to finish
        p_stress.join()

        # Shutdown logger
        if p_logger.is_alive():
            p_logger.terminate()
            p_logger.join()

        print("--- All processes complete ---")
