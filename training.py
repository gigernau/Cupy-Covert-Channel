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


# Funzione per calcolare i valori minimi e massimi globali su tutti i dati per ogni caratteristica
def calculate_global_min_max1(base_folder):
    min_vals = None
    max_vals = None
    
    # Itera su ogni cartella all'interno della cartella principale (ogni cartella rappresenta una classe)
    for class_folder in os.listdir(base_folder):
        class_path = os.path.join(base_folder, class_folder)
        
        # Verifica che sia una cartella
        if os.path.isdir(class_path):
            # Itera su ogni file CSV all'interno della cartella
            for file in glob.glob(os.path.join(class_path, "*.csv")):
                df = pd.read_csv(file)
                df = df.drop(columns=['Operation', 'Execution ID'], errors='ignore')
                
                # Calcolo del minimo e massimo globale
                if min_vals is None:
                    min_vals = df.min()
                    max_vals = df.max()
                else:
                    min_vals = pd.concat([min_vals, df.min()], axis=1).min(axis=1)
                    max_vals = pd.concat([max_vals, df.max()], axis=1).max(axis=1)
    
    if min_vals is None or max_vals is None:
        raise ValueError("Could not calculate global min and max values. Check input files.")
    
    return min_vals, max_vals

# Funzione per caricare e preparare i dati delle classi per l'addestramento (le classi sono ora cartelle)
def load_and_prepare_class_data_from_folders(base_folder, min_vals, max_vals):
    sequences = []
    labels = []
    
    # Itera su ogni cartella all'interno della cartella principale (ogni cartella rappresenta una classe)
    for class_folder in os.listdir(base_folder):
        class_path = os.path.join(base_folder, class_folder)
        
        # Verifica che sia una cartella
        if os.path.isdir(class_path):
            print(f"Processing class: {class_folder}")
            
            # Itera su ogni file CSV all'interno della cartella
            for file in glob.glob(os.path.join(class_path, "*.csv")):
                #print(f"Loading file: {file}")
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
                    
                    # Padding della sequenza se è più corta della lunghezza massima
                    if sequence_len < max_len:
                        padded_sequence = np.pad(exec_group_normalized, ((0, max_len - sequence_len), (0, 0)), mode='constant', constant_values=0)
                    else:
                        padded_sequence = exec_group_normalized[:max_len]
                    
                    sequences.append(padded_sequence)
                    labels.append(class_folder)  # La classe viene assegnata in base alla cartella
    
    print(f"Total sequences loaded: {len(sequences)}")
    return np.array(sequences), np.array(labels)

# Funzione per normalizzare i dati utilizzando i valori minimi e massimi globali
def normalize_data(df, min_vals, max_vals):
    if min_vals is None or max_vals is None:
        raise ValueError("min_vals and max_vals must be provided for normalization.")
    
    if df.shape[1] != len(min_vals) or df.shape[1] != len(max_vals):
        raise ValueError("Data shape does not match the shape of min_vals and max_vals.")
    
    # Normalizzazione: sottrazione del minimo e divisione per il range
    df_normalized = (df - min_vals) / (max_vals - min_vals)
    df_normalized.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_normalized.fillna(0, inplace=True)

    return df_normalized






# Funzione principale per addestrare il modello
def main():

    parser = argparse.ArgumentParser(description="Training ROCKET model.")
    parser.add_argument('--data', type=str, default='path_to_trained_data', help='Path to the trained data')
    parser.add_argument('--split', type=int, default=10, help='Split number')
    parser.add_argument('--op', type=str, nargs='+', default=['sort','linalg'], help='List of operations to be classified')
    args = parser.parse_args()

    folder_name = args.data # Cartella dei dati di addestramento
    #folder_name = 'dataset/train'  # Cartella dei dati di addestramento
    #class_names = ['matmul', 'dot', 'linalg', 'sort', 'transpose', 'sum', 'sin', 'exp', 'log', 'rand', 'mean', 'std']  # Nomi delle classi da riconoscere
    class_names = args.op  # Nomi delle classi da riconoscere

    # Calcolo dei valori minimi e massimi globali per normalizzare i dati
    min_vals, max_vals = calculate_global_min_max(folder_name, class_names)
    #min_vals, max_vals = calculate_global_min_max1(folder_name)

    all_sequences = []
    all_labels = []

    # Caricamento e preparazione dei dati di ogni classe
    for class_name in class_names:
        print(f"Processing class: {class_name}")
        sequences, labels = load_and_prepare_class_data(class_name, folder_name, min_vals, max_vals)

        all_sequences.extend(sequences)
        all_labels.extend(labels)
    
    print(f"Total sequences loaded: {len(all_sequences)}")
    
    # Caricamento e preparazione dei dati dalle cartelle
    #all_sequences, all_labels = load_and_prepare_class_data_from_folders(folder_name, min_vals, max_vals)

    if len(all_sequences) == 0 or len(all_labels) == 0:
        raise ValueError("No sequences or labels were loaded. Please check the input files.")

    X, y = prepare_data_for_training(all_sequences, all_labels)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Encoding delle etichette in valori numerici

    # Divisione del dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configurazione del pipeline: ROCKET + RidgeClassifierCV
    rocket = Rocket()
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

    model = make_pipeline(rocket, classifier)

    print("Training model with early stopping...")

    # Configurazione di GridSearchCV per l'early stopping e la cross-validation
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

    grid_search.fit(X_train, y_train)  # Addestramento del modello con GridSearchCV
    
    print("Evaluating model...")
    test_acc = grid_search.score(X_test, y_test)  # Valutazione del modello sul test set
    print(f"Test Accuracy: {test_acc:.4f}")

    #Salvataggio del miglior modello addestrato
    joblib.dump(grid_search.best_estimator_, f'model/{class_names}_{test_acc:.4f}.joblib')
    
    print(f"Model saved to 'model/{class_names}_{test_acc:.4f}.joblib'")

    # Salvataggio dei parametri di normalizzazione e del label encoder
    min_vals.to_csv(f'weights/min_vals_{class_names}.csv')
    max_vals.to_csv(f'weights/max_vals_{class_names}.csv')
    joblib.dump(label_encoder, f'weights/label_encoder_{class_names}.joblib')
    print("Normalization parameters and label encoder saved.")

    return min_vals, max_vals, label_encoder, test_acc, class_names,folder_name

def run_inference_on_files(model, label_encoder, min_vals, max_vals, folder_name):
    results = []
    correct_predictions = {}
    total_predictions = {}

    # Ottieni tutti i file CSV nella cartella e ordinali alfabeticamente
    file_paths = glob.glob(os.path.join(folder_name, "*.csv"))
    file_paths.sort()  # Ordina i file alfabeticamente

    for file_path in file_paths:
        # Prendi il nome della classe dal nome del file (prima parte prima dell'underscore)
        class_name = os.path.basename(file_path).split('_')[0]
        
        df = pd.read_csv(file_path)
        df = df.drop(columns=['Operation', 'Execution ID'], errors='ignore')
        
        exec_group_normalized = normalize_data(df, min_vals, max_vals)
        X = np.array([exec_group_normalized])
        
        y_pred = model.predict(X)
        predicted_label = label_encoder.inverse_transform(y_pred)[0]
        
        results.append((class_name, predicted_label, file_path))
        
        # Accumula le statistiche di predizione per ciascuna classe
        if class_name not in total_predictions:
            total_predictions[class_name] = 0
            correct_predictions[class_name] = 0
        
        total_predictions[class_name] += 1
        if class_name == predicted_label:
            correct_predictions[class_name] += 1
    
    return results, correct_predictions, total_predictions

def test_model_inference(model, label_encoder, min_vals, max_vals, folder_name):
    # Esegui le predizioni sui nuovi dati
    results, correct_predictions, total_predictions = run_inference_on_files(model, label_encoder, min_vals, max_vals, folder_name)
    
    # Riassunto delle predizioni
    print("\nRiassunto delle Predizioni:")
    overall_correct = 0
    overall_total = 0
    class_accuracies = {}

    for class_name in total_predictions:
        correct = correct_predictions.get(class_name, 0)
        total = total_predictions[class_name]
        accuracy = (correct / total) * 100
        class_accuracies[class_name] = accuracy  # Salva l'accuratezza per ogni classe
        print(f"Classe: {class_name}, Corrette: {correct}, Totali: {total}, Precisione: {accuracy:.2f}%")
        overall_correct += correct
        overall_total += total

    # Precisione complessiva
    if overall_total > 0:
        overall_accuracy = (overall_correct / overall_total) * 100
        print(f"\nPrecisione complessiva: {overall_accuracy:.2f}%")
    else:
        print("Nessuna predizione effettuata.")
    
    # Elenco delle percentuali di accuratezza per ogni classe
    print("\nPercentuali di Accuratezza per Classe:")
    for class_name, accuracy in class_accuracies.items():
        print(f"{class_name}: {accuracy:.2f}%")

    return class_accuracies  # Restituisci le percentuali per ogni classe





# Punto di ingresso principale: parsing degli argomenti e avvio del training e predizione
if __name__ == "__main__":
    
    min_vals, max_vals, label_encoder, test_acc, class_names, dataset = main()
    
    
    
    # Caricamento del modello addestrato
    model = joblib.load(f'model/{class_names}_{test_acc:.4f}.joblib')


    # # Test del modello sui nuovi dati
    # class_accuracies = test_model_inference(model, label_encoder, min_vals, max_vals, dataset)

    # print("\nPercentuali di accuratezza finale per ogni classe:")
    # for class_name, accuracy in class_accuracies.items():
    #     print(f"{class_name}: {accuracy:.2f}%")
