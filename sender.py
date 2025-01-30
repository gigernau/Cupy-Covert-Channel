import os
import glob
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import argparse

import cupy as cp
import numpy as np
import time
import sys
import nvidia_smi
from utils import * 

# Global matrices for CuPy operations
global a, b
a = cp.random.rand(4000, 4000)
b = cp.random.rand(4000, 4000)

# Define a binary alphabet
custom_alphabet = ['0', '1']
base = len(custom_alphabet)  # Base 2 for binary encoding


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Inference ROCKET model.")
    parser.add_argument('--message', type=str, default='OK', help='String to encode and send')
    parser.add_argument('--op', type=str, nargs='+', default=['sort', 'linalg'], help='List of operations to be classified')

    args = parser.parse_args()
    class_names = args.op  # Names of the classes to recognize

    # Mapping between binary symbols and operations
    operation_names = {
        '0': class_names[0],
        '1': class_names[1],
    }

    operation_names2 = {
        class_names[0]: '0',
        class_names[1]: '1',
    }

    # Start measuring total execution time
    start_total_time = time.time()

    # User input: secret message
    print(f"Secret message: {args.message} ")

    # Encode the message directly (it is pre-binary encoded here)
    encoded = args.message
    print(f"Message encoded in binary: {encoded}")

    # Convert binary encoding to corresponding operations
    operations = custom_alphabet_to_operations(encoded, operation_names)
    print(f"Operations: {operations}")

    # Configuration parameters
    N = len(operations)  # Number of operations
    folder_name = "message/"  # Directory to store output metrics
    update_interval_s = 0.0001  # GPU metrics sampling interval
    monitor_duration_s = 3  # Duration to monitor GPU metrics for each operation

    # Initialize the NVIDIA SMI library
    nvidia_smi.nvmlInit()

    # Create the directory if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Preliminary warmup with operations
    i = 0
    time.sleep(3)

    # Pre-run a few operations to stabilize performance
    for _ in range(5):
        csv_filename = f'{i + 1}.csv'
        a = cp.random.rand(4000, 4000) 
        b = cp.random.rand(4000, 4000)
        run_operation(a, b, i + 1, operations[1], monitor_duration_s, update_interval_s, folder_name, csv_filename, 0)

    print("Start operations")
    start_op_time_tot = time.time()  # Start measuring total operations time

    # Execute the operations in sequence
    for op in operations:
        csv_filename = f'{i + 1}.csv'
        a = cp.random.rand(4000, 4000)
        b = cp.random.rand(4000, 4000)
        time.sleep(0.5)

        # Measure the execution time for the operation
        start_op_time = time.time()
        run_operation(a, b, i + 1, op, monitor_duration_s, update_interval_s, folder_name, csv_filename, 1)
        op_time = time.time() - start_op_time
        print(f"Time for operation {op} (execution {i + 1}): {op_time:.4f} seconds")

        if (i + 1) < N:
            time.sleep(3)
        i += 1

    op_time_tot = time.time() - start_op_time_tot
    print(f"Total time for operations execution: {op_time_tot:.4f} seconds")


    # End of total time measurement
    total_time = time.time() - start_total_time
    print(f"Total execution time: {total_time:.4f} seconds")

    print(f"\n\n**************************************************")
    print(f"\nBits sent: {args.message}")
    print(f"\n**************************************************")



if __name__ == "__main__":
    main()