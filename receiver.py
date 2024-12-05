import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import argparse

from utils import *



# Definisci l'alfabeto personalizzato (può essere anche binario)
custom_alphabet = ['0', '1']  # O ['0', '1'] per il caso binario
base = len(custom_alphabet)



def binario_a_stringa(binario):
    # Divide la stringa binaria in blocchi di 8 bit
    caratteri = [binario[i:i+8] for i in range(0, len(binario), 8)]
    
    # Converte ogni blocco di 8 bit in un carattere
    stringa = ''.join([chr(int(car, 2)) for car in caratteri])
    
    return stringa



# Funzione principale dello script
def main():
    start_op_time_tot = time.time()

    parser = argparse.ArgumentParser(description="Inference ROCKET model.")
    parser.add_argument('--model', type=str, default='path_to_trained_model.joblib', help='Path to the trained model')
    parser.add_argument('--folder', type=str, default='metrics', help='Path to the folder with input files')
    parser.add_argument('--op', type=str, nargs='+', default=['sort','linalg'], help='List of operations to be classified')
    parser.add_argument('--test', type=str, help='STR send to test')
    
    args = parser.parse_args()
    class_names = args.op  # Nomi delle classi da riconoscere
    test_message = args.test

    operation_names = {
    '0': class_names[0],
    '1': class_names[1],

    }

    operation_names2 = {
        class_names[0]: '0',
        class_names[1]:'1',

    }


    model_path = args.model
    label_encoder_path = f'weights/label_encoder_{class_names}.joblib'
    folder_name = args.folder
    min_vals_file = f'weights/min_vals_{class_names}.csv'
    max_vals_file = f'weights/max_vals_{class_names}.csv'

    min_vals, max_vals = load_min_max_vals(min_vals_file, max_vals_file)
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    label_encoder = load_label_encoder(label_encoder_path)
    
    start_msg_time = time.time()
    results, correct_predictions, total_predictions = run_inference_on_files(model, label_encoder, min_vals, max_vals, folder_name)
    msg_time = time.time() - start_msg_time
    print(f"Tempo per la classificazione totale del messaggio {msg_time:.4f} secondi")

    # Stampa i risultati dell'inferenza
    for class_name, predicted_label, file_path in results:
        print(f"File: {file_path}, Attesa: {class_name}, Predizione: {predicted_label}")
    
    # Prepara le predizioni per la ricodifica
    ris = [predicted_label for _, predicted_label, _ in results]
        
    print(f"Total predictions: {ris}")

    # Converti le operazioni di nuovo in una stringa codificata
    new_encoded_string = operations_to_custom_alphabet(ris,operation_names)
    print(f"Custom string from operations: {new_encoded_string}")

    # Decodifica dalla nuova stringa codificata
    decoded_from_operations = custom_alphabet_to_string(new_encoded_string)
    print(f"Decoded from operations: {decoded_from_operations}")

    op_time_tot = time.time() - start_op_time_tot
    print(f"Tempo per l'esecuzione totale {op_time_tot:.4f} secondi")
    

    bin_str1 = test_message
    print(bin_str1)
    bin_str2 = str(new_encoded_string) # Stringa identica per test

    # Confronta le stringhe binarie
    matching_bits, differing_bits, accuracy, matching_bytes, total_bytes = compare_binary_strings(bin_str1, bin_str2)

    print(f"Bit uguali: {matching_bits}")
    print(f"Bit diversi: {differing_bits}")
    print(f"Accuratezza: {accuracy:.2f}%")
    print(f"Byte uguali: {matching_bytes} su {total_bytes}")

    # Converte il binario in stringa
    stringa = binario_a_stringa(bin_str2)
    print(f"Stringa risultante: {stringa}")

# Punto di ingresso dello script
if __name__ == "__main__":

    
    main()
    
