from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import StratifiedKFold
import joblib
from utils import calculate_global_min_max, load_and_prepare_class_data, prepare_data_for_training
import numpy as np
import argparse
import os
import glob
import pandas as pd


# Function to compute the global minimum and maximum values for each feature across all data
def calculate_global_min_max1(base_folder):
    min_vals = None
    max_vals = None
    
    # Iterate over each subfolder in the base folder (each subfolder represents a class)
    for class_folder in os.listdir(base_folder):
        class_path = os.path.join(base_folder, class_folder)
        
        # Check if it is a directory
        if os.path.isdir(class_path):
            # Iterate over each CSV file in the folder
            for file in glob.glob(os.path.join(class_path, "*.csv")):
                df = pd.read_csv(file)
                df = df.drop(columns=['Operation', 'Execution ID'], errors='ignore')
                
                # Calculate global minimum and maximum values
                if min_vals is None:
                    min_vals = df.min()
                    max_vals = df.max()
                else:
                    min_vals = pd.concat([min_vals, df.min()], axis=1).min(axis=1)
                    max_vals = pd.concat([max_vals, df.max()], axis=1).max(axis=1)
    
    if min_vals is None or max_vals is None:
        raise ValueError("Could not calculate global min and max values. Check input files.")
    
    return min_vals, max_vals

# Function to load and prepare class data for training (classes are now folders)
def load_and_prepare_class_data_from_folders(base_folder, min_vals, max_vals):
    sequences = []
    labels = []
    
    # Iterate over each subfolder in the base folder (each subfolder represents a class)
    for class_folder in os.listdir(base_folder):
        class_path = os.path.join(base_folder, class_folder)
        
        # Check if it is a directory
        if os.path.isdir(class_path):
            print(f"Processing class: {class_folder}")
            
            # Iterate over each CSV file in the folder
            for file in glob.glob(os.path.join(class_path, "*.csv")):
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
                    labels.append(class_folder)  # Assign the class based on the folder name
    
    print(f"Total sequences loaded: {len(sequences)}")
    return np.array(sequences), np.array(labels)

# Function to normalize data using global minimum and maximum values
def normalize_data(df, min_vals, max_vals):
    if min_vals is None or max_vals is None:
        raise ValueError("min_vals and max_vals must be provided for normalization.")
    
    if df.shape[1] != len(min_vals) or df.shape[1] != len(max_vals):
        raise ValueError("Data shape does not match the shape of min_vals and max_vals.")
    
    # Normalization: subtract the minimum and divide by the range
    df_normalized = (df - min_vals) / (max_vals - min_vals)
    df_normalized.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_normalized.fillna(0, inplace=True)

    return df_normalized


# Main function to train the model
def main():

    parser = argparse.ArgumentParser(description="Training ROCKET model.")
    parser.add_argument('--data', type=str, default='path_to_trained_data', help='Path to the training data')
    parser.add_argument('--split', type=int, default=10, help='Number of splits for cross-validation')
    parser.add_argument('--op', type=str, nargs='+', default=['sort','linalg'], help='List of operations to classify')
    args = parser.parse_args()

    folder_name = args.data  # Training data folder
    class_names = args.op  # Names of the classes to recognize

    # Compute global minimum and maximum values to normalize the data
    min_vals, max_vals = calculate_global_min_max(folder_name, class_names)

    all_sequences = []
    all_labels = []

    # Load and prepare data for each class
    for class_name in class_names:
        print(f"Processing class: {class_name}")
        sequences, labels = load_and_prepare_class_data(class_name, folder_name, min_vals, max_vals)

        all_sequences.extend(sequences)
        all_labels.extend(labels)
    
    print(f"Total sequences loaded: {len(all_sequences)}")
    
    if len(all_sequences) == 0 or len(all_labels) == 0:
        raise ValueError("No sequences or labels were loaded. Please check the input files.")

    X, y = prepare_data_for_training(all_sequences, all_labels)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Encode labels as numeric values

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configure the pipeline: ROCKET + RidgeClassifierCV
    rocket = Rocket()
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    model = make_pipeline(rocket, classifier)

    print("Training model with early stopping...")

    # Configure GridSearchCV for early stopping and cross-validation
    cv = StratifiedKFold(n_splits=args.split, shuffle=True, random_state=42)
    scorer = make_scorer(accuracy_score)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid={},
        scoring=scorer,
        cv=cv,
        verbose=3,
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X_train, y_train)  # Train the model using GridSearchCV
    
    print("Evaluating model...")
    test_acc = grid_search.score(X_test, y_test)  # Evaluate the model on the test set
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the best-trained model
    joblib.dump(grid_search.best_estimator_, f'model/{class_names}_{test_acc:.4f}.joblib')
    
    # Unisce i nomi delle classi in una stringa separata da "_"
    class_names= "_".join(class_names)

    #print(f"Model saved to 'model/{class_names}_{test_acc:.4f}.joblib'")
    print(f"Model saved to 'model/{class_names}.joblib'")

    # Save normalization parameters and the label encoder
    min_vals.to_csv(f'weights/min_vals_{class_names}.csv')
    max_vals.to_csv(f'weights/max_vals_{class_names}.csv')
    joblib.dump(label_encoder, f'weights/label_encoder_{class_names}.joblib')
    print("Normalization parameters and label encoder saved.")

    return min_vals, max_vals, label_encoder, test_acc, class_names, folder_name

if __name__ == "__main__":
    main()