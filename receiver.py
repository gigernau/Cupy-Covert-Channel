import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import argparse

from utils import *


# Define a custom alphabet (binary in this case)
custom_alphabet = ['0', '1']  # Use ['0', '1'] for binary encoding
base = len(custom_alphabet)


# Function to convert a binary string to a text string
def binary_to_string(binary):
    # Divide the binary string into blocks of 8 bits
    characters = [binary[i:i+8] for i in range(0, len(binary), 8)]
    
    # Convert each 8-bit block to a character
    string = ''.join([chr(int(char, 2)) for char in characters])
    
    return string


# Main function of the script
def main():
    start_op_time_tot = time.time()

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Inference ROCKET model.")
    parser.add_argument('--model', type=str, default='path_to_trained_model.joblib', help='Path to the trained model')
    parser.add_argument('--folder', type=str, default='metrics', help='Path to the folder with input files')
    parser.add_argument('--op', type=str, nargs='+', default=['sort', 'linalg'], help='List of operations to be classified')
    parser.add_argument('--test', type=str, help='Binary string to test')
    
    args = parser.parse_args()
    class_names = args.op  # Names of the classes to recognize
    test_message = args.test

    # Mapping between binary symbols and operations
    operation_names = {
        '0': class_names[0],
        '1': class_names[1],
    }

    operation_names2 = {
        class_names[0]: '0',
        class_names[1]: '1',
    }

    # Paths to model and auxiliary files
    model_path = args.model
    label_encoder_path = f'weights/label_encoder_{class_names}.joblib'
    folder_name = args.folder
    min_vals_file = f'weights/min_vals_{class_names}.csv'
    max_vals_file = f'weights/max_vals_{class_names}.csv'

    # Load normalization parameters and model
    min_vals, max_vals = load_min_max_vals(min_vals_file, max_vals_file)
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    label_encoder = load_label_encoder(label_encoder_path)
    
    # Perform inference on all files in the specified folder
    start_msg_time = time.time()
    results, correct_predictions, total_predictions = run_inference_on_files(model, label_encoder, min_vals, max_vals, folder_name)
    msg_time = time.time() - start_msg_time
    print(f"Total time for message classification: {msg_time:.4f} seconds")

    # Print inference results
    for class_name, predicted_label, file_path in results:
        print(f"File: {file_path}, Expected: {class_name}, Predicted: {predicted_label}")
    
    # Prepare predictions for re-encoding
    predicted_classes = [predicted_label for _, predicted_label, _ in results]
        
    print(f"Total predictions: {predicted_classes}")

    # Convert operations back to a custom encoded string
    new_encoded_string = operations_to_custom_alphabet(predicted_classes, operation_names)
    
    print(f"\n\n**************************************************")
    print(f"\nBits received: {new_encoded_string}")
    print(f"\n**************************************************")

    # Total execution time
    op_time_tot = time.time() - start_op_time_tot
    print(f"\nTotal execution time: {op_time_tot:.4f} seconds")
    
    # Compare the test message with the newly decoded message
    bin_str1 = test_message  # Original binary string for testing
    bin_str2 = str(new_encoded_string)  # Binary string obtained from decoding operations

    # Compare binary strings
    matching_bits, differing_bits, accuracy, matching_bytes, total_bytes = compare_binary_strings(bin_str1, bin_str2)

    print(f"Matching bits: {matching_bits}")
    print(f"Differing bits: {differing_bits}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Matching bytes: {matching_bytes} out of {total_bytes}")


# Script entry point
if __name__ == "__main__":
    main()