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
import h5py
import pickle
import dill
import chardet
import io
import onnx
import onnxruntime as ort




# Definisci l'alfabeto personalizzato di 6 simboli
custom_alphabet = ['0', '1']
base = len(custom_alphabet)  # Base del nostro sistema di codifica


# Funzione per calcolare i valori minimi e massimi globali su tutti i dati per ogni caratteristica
def calculate_global_min_max(folder_name, class_names):
    min_vals = None
    max_vals = None
    
    # Loop su ogni classe e su ogni file CSV corrispondente
    for class_name in class_names:
        for file in glob.glob(os.path.join(folder_name, f"{class_name}_*.csv")):
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

# Funzione per normalizzare i dati utilizzando i valori minimi e massimi globali
def normalize_data(df, min_vals, max_vals):
    if min_vals is None or max_vals is None:
        raise ValueError("min_vals and max_vals must be provided for normalization.")
    
    if df.shape[1] != len(min_vals) or df.shape[1] != len(max_vals):
        raise ValueError("Data shape does not match the shape of min_vals and max_vals.")
    
    # Normalizzazione: sottrazione del minimo e divisione per il range
    df_normalized = (df - min_vals) / (max_vals - min_vals)
    df_normalized.replace([cp.inf, -cp.inf], cp.nan, inplace=True)
    df_normalized.fillna(0, inplace=True)

    return df_normalized

# Funzione per caricare e preparare le sequenze di dati per l'inferenza (predizione)
def load_and_prepare_sequences_for_inference(folder_name, min_vals, max_vals):
    sequences = []
    file_names = []
    
    # Loop su ogni file nella cartella specificata
    for file in glob.glob(os.path.join(folder_name, "*.csv")):
        #print(f"Loading file: {file}")
        df = pd.read_csv(file)
        df = df.drop(columns=['Operation', 'Execution ID'], errors='ignore')
        
        # Normalizzazione dei dati
        exec_group_normalized = normalize_data(df, min_vals, max_vals)
        
        sequence_len = len(exec_group_normalized)
        max_len = len(min_vals)

        # Padding della sequenza se è più corta della lunghezza massima
        if sequence_len < max_len:
            padded_sequence = cp.pad(exec_group_normalized, ((0, max_len - sequence_len), (0, 0)), mode='constant', constant_values=0)
        else:
            padded_sequence = exec_group_normalized[:max_len]
        
        sequences.append(padded_sequence)
        file_names.append(os.path.basename(file))
    
    return np.array(sequences), file_names

# Funzione per caricare e preparare i dati delle classi per l'addestramento
def load_and_prepare_class_data(class_name, folder_name, min_vals, max_vals):
    sequences = []
    labels = []
    
    # Loop su ogni file CSV corrispondente alla classe specificata
    for file in glob.glob(os.path.join(folder_name, f"{class_name}_*.csv")):
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
            labels.append(class_name)
    
    print(f"Loaded {len(sequences)} sequences for class {class_name}")
    return np.array(sequences), np.array(labels)

# Funzione per preparare i dati per l'addestramento: trasformazione in array numpy
def prepare_data_for_training(sequences, labels):
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"Prepared data with shape: X={X.shape}, y={y.shape}")
    
    return X, y


# Funzione per effettuare predizioni su nuovi dati utilizzando il modello addestrato
def predict_on_new_data(model, folder_name, min_vals, max_vals, label_encoder):
    all_predictions = []

    # Loop su ogni file di dati nella cartella specificata
    for file in glob.glob(os.path.join(folder_name, "*.csv")):
        print(f"\nProcessing file: {file}")
        df = pd.read_csv(file)
        
        # if 'Execution ID' in df.columns:
        #     execution_groups = df.groupby('Execution ID')
        # else:
        #     execution_groups = [(None, df)]

        # Preparazione di ogni gruppo di esecuzione
        for _, exec_group in execution_groups:
            exec_group = exec_group.drop(columns=['Operation', 'Execution ID'], errors='ignore')
            
            exec_group_normalized = normalize_data(exec_group, min_vals, max_vals)
            
            sequence_len = len(exec_group_normalized)
            max_len = len(min_vals)

            # Padding della sequenza se è più corta della lunghezza massima
            if sequence_len < max_len:
                padded_sequence = cp.pad(exec_group_normalized, ((0, max_len - sequence_len), (0, 0)), mode='constant', constant_values=0)
            else:
                padded_sequence = exec_group_normalized[:max_len]

            X = cp.array([padded_sequence])
            y_pred = model.predict(X)  # Predizione della classe
            predicted_label = label_encoder.inverse_transform(y_pred)[0]
            
            all_predictions.append((file, predicted_label))
            
    return all_predictions





#####################################
#INFERENCE FUCTION

# Funzione per caricare i valori minimi e massimi dai file CSV
def load_min_max_vals(min_vals_file, max_vals_file):
    if not os.path.exists(min_vals_file) or not os.path.exists(max_vals_file):
        raise FileNotFoundError(f"Min or max values files not found: {min_vals_file}, {max_vals_file}")
    
    min_vals = pd.read_csv(min_vals_file, index_col=0).squeeze()
    max_vals = pd.read_csv(max_vals_file, index_col=0).squeeze()
    
    if isinstance(min_vals, pd.DataFrame):
        min_vals = min_vals.squeeze()
    if isinstance(max_vals, pd.DataFrame):
        max_vals = max_vals.squeeze()
    
    return min_vals, max_vals

# Funzione per caricare il LabelEncoder
def load_label_encoder(label_encoder_file):
    if not os.path.exists(label_encoder_file):
        raise FileNotFoundError(f"Label encoder file not found: {label_encoder_file}")
    
    label_encoder = joblib.load(label_encoder_file)
    return label_encoder


def run_inference_on_files(model, label_encoder, min_vals, max_vals, folder_name):
    results = []
    correct_predictions = {}
    total_predictions = {}

    # Ottieni tutti i file CSV e ordinali in base al numero nel nome del file
    file_paths = glob.glob(os.path.join(folder_name, "*.csv"))
    file_paths.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

    for file_path in file_paths:
        class_name = os.path.basename(file_path).split('_')[0]
        
        df = pd.read_csv(file_path)
        df = df.drop(columns=['Operation', 'Execution ID'], errors='ignore')
        
        exec_group_normalized = normalize_data(df, min_vals, max_vals)
        X = np.array([exec_group_normalized])
        
        #start_msgclass_time = time.time()
        y_pred = model.predict(X)
        #msg_time = time.time() - start_msgclass_time
        #print(f"Tempo per la classificazione di {class_name}: {msg_time:.4f} s")
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


#####################################
#MAIN



# Funzione per convertire un numero in una rappresentazione dell'alfabeto personalizzato
def num_to_custom_symbol(num):
    if num == 0:
        return custom_alphabet[0]
    result = []
    while num > 0:
        result.append(custom_alphabet[num % base])
        num //= base
    return ''.join(reversed(result))

# Funzione per convertire una rappresentazione dell'alfabeto personalizzato in un numero
def custom_symbol_to_num(symbols):
    num = 0
    for symbol in symbols:
        num = num * base + custom_alphabet.index(symbol)
    return num


# Codifica una stringa in binario
def string_to_custom_alphabet(input_string):
    encoded = ''
    for char in input_string:
        ascii_value = ord(char)  # Ottieni il valore ASCII del carattere
        binary_value = format(ascii_value, '08b')  # Converte il valore ASCII in una stringa binaria di 8 bit
        encoded += binary_value  # Aggiungi il valore binario alla stringa codificata
    return encoded

# Decodifica una stringa dall'alfabeto personalizzato
def custom_alphabet_to_string(encoded_string):
    decoded = ''
    while encoded_string:
        part = encoded_string[:3]  # Usa triplette
        num_value = custom_symbol_to_num(part)
        decoded += chr(num_value)
        encoded_string = encoded_string[3:]  # Rimuovi le triplette elaborate
    return decoded

# Converti una stringa codificata in una lista di operazioni
def custom_alphabet_to_operations(encoded_string,operation_names):
    operations = []
    for symbol in encoded_string:
        if symbol in operation_names:
            operations.append(operation_names[symbol])
        else:
            operations.append("UNKNOWN")  # Gestisci simboli sconosciuti
    return operations

# Converti una lista di operazioni in una stringa codificata
def operations_to_custom_alphabet(operations,operation_names):
    reverse_op_map = {v: k for k, v in operation_names.items()}
    encoded_string = ''
    for op in operations:
        symbol = reverse_op_map.get(op, 'A')  # Usa 'A' come default per operazioni sconosciute
        encoded_string += symbol
    return encoded_string


# Funzione per salvare i dati su file CSV
def save_to_csv(data, folder_name, csv_filename):
    file_exists = os.path.isfile(os.path.join(folder_name, csv_filename))
    try:
        with open(os.path.join(folder_name, csv_filename), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                # Scrivi l'intestazione solo se il file non esiste
                writer.writerow(['Execution ID', 'Milliseconds Passed', 'Temperature (C)', 'Power Usage (W)', 
                                 'GPU Utilization (%)', 'Memory Usage (MB)', 'Cache Usage', 'Clocks Video (MHz)'])
            # Scrivi le metriche raccolte
            writer.writerows(data)
    except Exception as e:
        print(f"Errore durante il salvataggio dei dati nel file {csv_filename}: {e}")

# Funzione per raccogliere le metriche della GPU
def collect_gpu_metrics(execution_id, monitor_duration_s, update_interval_s, folder_name, csv_filename):
    start_time = time.time()  # Tempo di inizio della raccolta
    metrics = []

    while time.time() - start_time < monitor_duration_s:
        try:
            # Esegui il comando nvidia-smi per ottenere le metriche della GPU
            result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,utilization.memory,clocks.current.video', '--format=csv,noheader,nounits'],
                                    stdout=subprocess.PIPE, text=True, check=True)
            gpu_info = result.stdout.strip().split(',')
            gpu_info = [round(float(x), 5) for x in gpu_info]

            # Calcola i millisecondi passati dall'inizio della raccolta
            milliseconds_passed = int((time.time() - start_time) * 1000)
            # Aggiungi le metriche alla lista
            metrics.append([execution_id, milliseconds_passed] + gpu_info)

            time.sleep(update_interval_s)  # Attendi per la durata dell'intervallo di aggiornamento

        except Exception as e:
            print(f"Errore durante la raccolta delle metriche: {e}")

    # Salva le metriche raccolte nel CSV
    save_to_csv(metrics, folder_name, csv_filename)

# Funzione per eseguire l'operazione specificata
def run_operation(a,b,execution_id, operation, monitor_duration_s, update_interval_s, folder_name, csv_filename,flag):
   
    if flag == 1:
       # Avvia la raccolta delle metriche in un thread separato
        
        # Esegui l'operazione specificata con CuPy
        try:
            metrics_thread = threading.Thread(target=collect_gpu_metrics, args=(execution_id, monitor_duration_s, update_interval_s, folder_name, csv_filename))
            metrics_thread.start()
            print(f"Operation: {operation}")

            if operation == 'linalg.solve':
                cp.linalg.solve(a,b)  # Determinante di una matrice
            elif operation == 'sort':
                cp.sort(a)  # Ordinamento di un array
            elif operation == 'transpose':
                cp.transpose(a)  # trasposta di un array
            elif operation == 'sum':
                cp.sum(a)  # trasposta di un array
            elif operation == 'rand':
                cp.random.rand(100)  # radice quadrata di un array
            elif operation == 'std':
                cp.std(a)  # radice quadrata di un array
            elif operation == 'matmul':
                cp.matmul(a,b)  # radice quadrata di un array
            
        except Exception as e:
            print(f"Errore durante l'esecuzione dell'operazione: {e}")

        # Attendi che la raccolta delle metriche sia completata
        metrics_thread.join()
    else:
         # Esegui l'operazione specificata con CuPy
        try:
            if operation == 'linalg.solve':
                cp.linalg.solve(a,b)  # Determinante di una matrice
            elif operation == 'sort':
                cp.sort(a)  # Ordinamento di un array
            elif operation == 'transpose':
                cp.transpose(a)  # trasposta di un array
            elif operation == 'sum':
                cp.sum(a)  # trasposta di un array
            elif operation == 'rand':
                cp.random.rand(100)  # radice quadrata di un array
            elif operation == 'std':
                cp.std(a)  # radice quadrata di un array
            
        except Exception as e:
            print(f"Errore durante l'esecuzione dell'operazione: {e}")

# Funzione principale dello script



def load_model(model):
    if os.path.exists(model):
        model = joblib.load(model)

    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    return model








    ####
    #RECEIVEr


def compare_binary_strings(bin_str1, bin_str2):
    # Assicurati che entrambe le stringhe abbiano la stessa lunghezza, in caso contrario, troncala alla più corta
    min_len = min(len(bin_str1), len(bin_str2))
    bin_str1 = bin_str1[:min_len]
    bin_str2 = bin_str2[:min_len]

    # Conta bit uguali e diversi
    matching_bits = sum(1 for b1, b2 in zip(bin_str1, bin_str2) if b1 == b2)
    differing_bits = len(bin_str1) - matching_bits

    # Calcola l'accuratezza
    accuracy = (matching_bits / len(bin_str1)) * 100

    # Divide in gruppi di 8 bit per verificare i byte
    byte_str1 = [bin_str1[i:i+8] for i in range(0, len(bin_str1), 8)]
    byte_str2 = [bin_str2[i:i+8] for i in range(0, len(bin_str2), 8)]

    # Conta byte uguali e diversi
    matching_bytes = sum(1 for b1, b2 in zip(byte_str1, byte_str2) if b1 == b2)
    total_bytes = min(len(byte_str1), len(byte_str2))

    return matching_bits, differing_bits, accuracy, matching_bytes, total_bytes