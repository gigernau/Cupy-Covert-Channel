import glob
import os
import pandas as pd
import cupy as cp
import numpy as np
import threading
import time
import subprocess
import csv
import sys
import nvidia_smi
import joblib

# Define a custom alphabet of two symbols (binary system)
custom_alphabet = ['0', '1']
base = len(custom_alphabet)  # Base of the encoding system


# Function to compute global minimum and maximum values for all data for each feature
def calculate_global_min_max(folder_name, class_names):
    min_vals = None
    max_vals = None
    
    # Loop through each class and its corresponding CSV files
    for class_name in class_names:
        for file in glob.glob(os.path.join(folder_name, f"{class_name}_*.csv")):
            df = pd.read_csv(file)
            df = df.drop(columns=['Operation', 'Execution ID'], errors='ignore')
            
            # Calculate global min and max values
            if min_vals is None:
                min_vals = df.min()
                max_vals = df.max()
            else:
                min_vals = pd.concat([min_vals, df.min()], axis=1).min(axis=1)
                max_vals = pd.concat([max_vals, df.max()], axis=1).max(axis=1)
    
    if min_vals is None or max_vals is None:
        raise ValueError("Could not calculate global min and max values. Check input files.")
    
    return min_vals, max_vals

# Function to normalize data using global min and max values
def normalize_data(df, min_vals, max_vals):
    if min_vals is None or max_vals is None:
        raise ValueError("min_vals and max_vals must be provided for normalization.")
    
    if df.shape[1] != len(min_vals) or df.shape[1] != len(max_vals):
        raise ValueError("Data shape does not match the shape of min_vals and max_vals.")
    
    # Normalization: subtract the min and divide by the range
    df_normalized = (df - min_vals) / (max_vals - min_vals)
    df_normalized.replace([cp.inf, -cp.inf], cp.nan, inplace=True)
    df_normalized.fillna(0, inplace=True)

    return df_normalized

# Function to load and prepare data sequences for inference (prediction)
def load_and_prepare_sequences_for_inference(folder_name, min_vals, max_vals):
    sequences = []
    file_names = []
    
    # Loop through each file in the specified folder
    for file in glob.glob(os.path.join(folder_name, "*.csv")):
        df = pd.read_csv(file)
        df = df.drop(columns=['Operation', 'Execution ID'], errors='ignore')
        
        # Normalize data
        exec_group_normalized = normalize_data(df, min_vals, max_vals)
        
        sequence_len = len(exec_group_normalized)
        max_len = len(min_vals)

        # Pad the sequence if it is shorter than the maximum length
        if sequence_len < max_len:
            padded_sequence = cp.pad(exec_group_normalized, ((0, max_len - sequence_len), (0, 0)), mode='constant', constant_values=0)
        else:
            padded_sequence = exec_group_normalized[:max_len]
        
        sequences.append(padded_sequence)
        file_names.append(os.path.basename(file))
    
    return np.array(sequences), file_names

# Function to load and prepare class data for training
def load_and_prepare_class_data(class_name, folder_name, min_vals, max_vals):
    sequences = []
    labels = []
    
    # Loop through each CSV file corresponding to the specified class
    for file in glob.glob(os.path.join(folder_name, f"{class_name}_*.csv")):
        df = pd.read_csv(file)
        
        if 'Execution ID' not in df.columns:
            print(f"'Execution ID' not found in {file}. Available columns: {df.columns}")
            continue
        
        execution_groups = df.groupby('Execution ID')
        for _, exec_group in execution_groups:
            exec_group = exec_group.drop(columns=['Operation', 'Execution ID'], errors='ignore')
            exec_group_normalized = normalize_data(exec_group, min_vals, max_vals)
            
            sequence_len = len(exec_group_normalized)
            max_len = len(min_vals)

            # Pad the sequence if it is shorter than the maximum length
            if sequence_len < max_len:
                padded_sequence = np.pad(exec_group_normalized, ((0, max_len - sequence_len), (0, 0)), mode='constant', constant_values=0)
            else:
                padded_sequence = exec_group_normalized[:max_len]
            
            sequences.append(padded_sequence)
            labels.append(class_name)
    
    print(f"Loaded {len(sequences)} sequences for class {class_name}")
    return np.array(sequences), np.array(labels)

# Function to prepare data for training: transform into NumPy arrays
def prepare_data_for_training(sequences, labels):
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"Prepared data with shape: X={X.shape}, y={y.shape}")
    
    return X, y


# Function to make predictions on new data using the trained model
def predict_on_new_data(model, folder_name, min_vals, max_vals, label_encoder):
    all_predictions = []

    # Loop through each data file in the specified folder
    for file in glob.glob(os.path.join(folder_name, "*.csv")):
        print(f"\nProcessing file: {file}")
        df = pd.read_csv(file)

        execution_groups = df.groupby('Execution ID') if 'Execution ID' in df.columns else [(None, df)]

        # Prepare each execution group
        for _, exec_group in execution_groups:
            exec_group = exec_group.drop(columns=['Operation', 'Execution ID'], errors='ignore')
            
            exec_group_normalized = normalize_data(exec_group, min_vals, max_vals)
            
            sequence_len = len(exec_group_normalized)
            max_len = len(min_vals)

            # Pad the sequence if it is shorter than the maximum length
            if sequence_len < max_len:
                padded_sequence = cp.pad(exec_group_normalized, ((0, max_len - sequence_len), (0, 0)), mode='constant', constant_values=0)
            else:
                padded_sequence = exec_group_normalized[:max_len]

            X = cp.array([padded_sequence])
            y_pred = model.predict(X)  # Predict the class
            predicted_label = label_encoder.inverse_transform(y_pred)[0]
            
            all_predictions.append((file, predicted_label))
            
    return all_predictions