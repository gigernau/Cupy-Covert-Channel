import os
import glob
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import argparse


import os
import cupy as cp
import numpy as np
import time

import sys
import nvidia_smi
from utils import * 


global a,b
a = cp.random.rand(4000, 4000)
b = cp.random.rand(4000, 4000)


# Definisci l'alfabeto binario
custom_alphabet = ['0', '1']
base = len(custom_alphabet)  # Base 2 per la codifica binaria



def main():
    parser = argparse.ArgumentParser(description="Inference ROCKET model.")
    parser.add_argument('--model', type=str, default='path_to_trained_model.joblib', help='Path to the trained model')
    parser.add_argument('--message', type=str, default='OK', help='Stringa')
    parser.add_argument('--op', type=str, nargs='+', default=['sort','linalg'], help='List of operations to be classified')

    args = parser.parse_args()
    class_names = args.op  # Nomi delle classi da riconoscere

    operation_names = {
    '0': class_names[0],
    '1': class_names[1],

    }

    operation_names2 = {
        class_names[0]: '0',
        class_names[1]:'1',

    }


    # Inizio del conteggio del tempo totale
    start_total_time = time.time()

    # Input dell'utente
    print(f"Messaggio segreto: {args.message} ")

    # # Codifica
    #encoded = string_to_custom_alphabet(args.message)
    
    encoded = args.message
    print(f"Stringa codificata in binario: {encoded}")
    # Converti in operazioni
    operations = custom_alphabet_to_operations(encoded,operation_names)
    print(f"Operazioni: {operations}")

    # Parametri di configurazione
    N = len(operations)
    folder_name = "message/"
    update_interval_s = 0.0001
    monitor_duration_s = 3

    # Inizializza la libreria nvidia_smi
    nvidia_smi.nvmlInit()

    # Crea la cartella se non esiste
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    i = 0
    time.sleep(3)
    # a = cp.random.rand(4000, 4000)
    # b = cp.random.rand(4000, 4000)
    # for _ in range(10):
    #     a = cp.sort(a) 
    #     a = cp.linalg.solve(a, b)

    for _ in range(5):
        csv_filename = f'{i + 1}.csv'
        a = cp.random.rand(4000, 4000) 
        b = cp.random.rand(4000, 4000)
        run_operation(a, b, i + 1, operations[1], monitor_duration_s, update_interval_s, folder_name, csv_filename,0)


    print("Start op")
    # Misura tempo di esecuzione per l'operazione
    start_op_time_tot = time.time()
    
    for op in operations:
        csv_filename = f'{i + 1}.csv'
        a = cp.random.rand(4000, 4000)
        b = cp.random.rand(4000, 4000)
        #print(f"Starting operation {op} execution {i + 1}")
        time.sleep(0.5)


        # Misura tempo di esecuzione per l'operazione
        start_op_time = time.time()
        run_operation(a, b, i + 1, op, monitor_duration_s, update_interval_s, folder_name, csv_filename, 1)
        op_time = time.time() - start_op_time
        print(f"Tempo per l'operazione {op} (esecuzione {i + 1}): {op_time:.4f} secondi")

        if (i + 1) < N:
            time.sleep(3)
        i += 1

    op_time_tot = time.time() - start_op_time_tot
    print(f"Tempo per l'esecuzione delle operazioni totale {op_time_tot:.4f} secondi")

    model_path = args.model
    label_encoder_path = f'weights/label_encoder_{class_names}.joblib'
    folder_name = 'message/'
    min_vals_file = f'weights/min_vals_{class_names}.csv'
    max_vals_file = f'weights/max_vals_{class_names}.csv'

    min_vals, max_vals = load_min_max_vals(min_vals_file, max_vals_file)

    # Fine conteggio del tempo totale
    total_time = time.time() - start_total_time
    print(f"Tempo totale di esecuzione: {total_time:.4f} secondi")

if __name__ == "__main__":
    main()











