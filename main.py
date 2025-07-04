import re
import time
import pandas as pd
import numpy as np
import os
import sys
from mlp.mlp import MLP
from mlp.utils import load_data, plot_learning_curves, Colors
from mlp.preprocessor import Preprocessor
from visualize.describe import describe
from visualize.pair_plot import plot_pair
from visualize.correlation_plot import plot_correlation
from visualize.results_plot import plot_seed_search, plot_prediction_results

CONFIG = {
    "dataset_path": None,
    "split": {
        "val_size": 0.2,  # 20% split for validation
        "seed_mode": "auto", # Will iterate and find the best seed through 10 iterations
        "manual_seed": 42 # Default seed for manual mode
    },
    "model": {
        "lr": 0.01, # Learning rate for the optimizer
        "epochs": 500,    # Number of iterations for training
        "batch_size": 32,
        "patience": 10,     # Threshold for early stopping
        "architecture": [24, 24], # Default architecture with two hidden layers of 24 neurons each
        "activations": ["relu", "relu"] # Default activations for the hidden layers (sigmoid is considered legacy)
    }
}
MODEL_PATH = 'saved_models/mlp_model.npz'
PRED_PATH = 'predictions.csv'
LOG_PATH = 'training_log.txt'

def cleanup_files(files_to_remove):
    """Removes a list of temporary files."""
    print(f"\n{Colors.CYAN}--- Cleaning up temporary files... ---{Colors.NC}")
    for f in files_to_remove:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"  - Removed {f}")
            except OSError as e:
                print(f"{Colors.RED}Error removing file {f}: {e}{Colors.NC}")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(r"""
███    ███ ██    ██ ██      ████████ ██ ██       █████  ██    ██ ███████ ██████     
████  ████ ██    ██ ██         ██    ██ ██      ██   ██  ██  ██  ██      ██   ██    
██ ████ ██ ██    ██ ██         ██    ██ ██      ███████   ████   █████   ██████     
██  ██  ██ ██    ██ ██         ██    ██ ██      ██   ██    ██    ██      ██   ██    
██      ██  ██████  ███████    ██    ██ ███████ ██   ██    ██    ███████ ██   ██    
                                                                                    
                                                                                    
██████  ███████ ██████   ██████ ███████ ██████  ████████ ██████   ██████  ███    ██ 
██   ██ ██      ██   ██ ██      ██      ██   ██    ██    ██   ██ ██    ██ ████   ██ 
██████  █████   ██████  ██      █████   ██████     ██    ██████  ██    ██ ██ ██  ██ 
██      ██      ██   ██ ██      ██      ██         ██    ██   ██ ██    ██ ██  ██ ██ 
██      ███████ ██   ██  ██████ ███████ ██         ██    ██   ██  ██████  ██   ████ 
""")

def validate_csv_file(path):
    """
    Validates that a file is a properly formatted CSV for the ML pipeline.
    Returns (is_valid, error_message, sample_count, feature_count)
    """
    try:
        # Check file extension
        if not path.lower().endswith('.csv'):
            return False, "File must have a .csv extension.", 0, 0
            
        # Try to load and validate the data structure
        df = pd.read_csv(path, header=None)
        
        # Check if we have enough columns (should be 31 or 32)
        if df.shape[1] < 31:
            return False, f"CSV file must have at least 31 columns. Found {df.shape[1]} columns.", 0, 0
            
        # If 32 columns, drop the first (ID column)
        if df.shape[1] == 32:
            df = df.drop(columns=[0])
        
        # Check if first column contains valid labels ('M' or 'B')
        y_categorical = df.iloc[:, 0]
        valid_labels = set(['M', 'B'])
        unique_labels = set(y_categorical.dropna().unique())
        
        if not unique_labels.issubset(valid_labels):
            return False, f"First column must contain only 'M' and 'B' labels. Found: {unique_labels}", 0, 0
            
        # Check if we have any data rows
        if df.shape[0] == 0:
            return False, "CSV file is empty.", 0, 0
            
        return True, "", df.shape[0], df.shape[1]-1
        
    except pd.errors.ParserError as e:
        return False, f"Invalid CSV format. {e}", 0, 0
    except Exception as e:
        return False, f"Error reading CSV file: {e}", 0, 0

def get_dataset_path():
    while True:
        clear_screen()
        path = input("Please enter the path to the dataset CSV file to be used for training: ")
        if not os.path.exists(path):
            print(f"{Colors.RED}File not found at '{path}'. Please try again.{Colors.NC}")
            input("Press Enter to continue...")
            continue
            
        # Validate that it's a proper CSV file
        is_valid, error_msg, sample_count, feature_count = validate_csv_file(path)
        
        if not is_valid:
            print(f"{Colors.RED}Error: {error_msg}{Colors.NC}")
            input("Press Enter to continue...")
            continue
            
        print(f"{Colors.GREEN}✓ Valid CSV file detected with {sample_count} samples and {feature_count} features.{Colors.NC}")
        CONFIG["dataset_path"] = path
        input("Press Enter to continue...")
        return

def configure_parameters_menu():
    while True:
        clear_screen()
        print(f"{Colors.BOLD}Configure Model & Split Parameters{Colors.NC}")
        print("-" * 40)
        print(f"{Colors.CYAN}--- Model Parameters ---{Colors.NC}")
        print(f"1. Learning Rate: {Colors.YELLOW}{CONFIG['model']['lr']}{Colors.NC}")
        print(f"2. Epochs: {Colors.YELLOW}{CONFIG['model']['epochs']}{Colors.NC}")
        print(f"3. Batch Size: {Colors.YELLOW}{CONFIG['model']['batch_size']}{Colors.NC}")
        print(f"4. Early Stopping Patience: {Colors.YELLOW}{CONFIG['model']['patience']}{Colors.NC}")
        print(f"5. Hidden Layers Architecture: {Colors.YELLOW}{CONFIG['model']['architecture']}{Colors.NC}")
        print(f"6. Hidden Layers Activations: {Colors.YELLOW}{CONFIG['model']['activations']}{Colors.NC}")
        print(f"{Colors.CYAN}--- Split Parameters ---{Colors.NC}")
        print(f"7. Validation Set Size (%): {Colors.YELLOW}{CONFIG['split']['val_size'] * 100:.0f}%{Colors.NC}")
        print(f"8. Seed Mode: {Colors.YELLOW}{CONFIG['split']['seed_mode']}{Colors.NC}")
        if CONFIG['split']['seed_mode'] == 'manual':
            print(f"   - Manual Seed: {Colors.YELLOW}{CONFIG['split']['manual_seed']}{Colors.NC}")
        print(f"m. {Colors.RED}Back to Main Menu{Colors.NC}")
        
        choice = input("Enter your choice: ").lower()
        try:
            if choice == '1': CONFIG['model']['lr'] = float(input("Enter new learning rate: "))
            elif choice == '2': CONFIG['model']['epochs'] = int(input("Enter number of epochs: "))
            elif choice == '3': CONFIG['model']['batch_size'] = int(input("Enter batch size: "))
            elif choice == '4': CONFIG['model']['patience'] = int(input("Enter patience epochs (0 to disable): "))
            elif choice == '5':
                arch_str = input("Enter hidden layer sizes (e.g., '24 24 12'): ")
                CONFIG['model']['architecture'] = [int(s) for s in arch_str.split()]
                if len(CONFIG['model']['architecture']) != len(CONFIG['model']['activations']):
                    CONFIG['model']['activations'] = ['relu'] * len(CONFIG['model']['architecture'])
                    print(f"{Colors.YELLOW}Activations reset to match new architecture. Please re-configure if needed.{Colors.NC}")
            elif choice == '6':
                if not CONFIG['model']['architecture']:
                    print(f"{Colors.RED}Please set architecture first.{Colors.NC}")
                    time.sleep(1)
                else:
                    new_activations = []
                    for i, size in enumerate(CONFIG['model']['architecture']):
                        act = input(f"Activation for hidden layer {i+1} (size {size}) ['relu' or 'sigmoid']: ").lower()
                        if act in ['relu', 'sigmoid']: new_activations.append(act)
                        else:
                            print(f"{Colors.RED}Invalid activation. Defaulting to 'relu'.{Colors.NC}")
                            new_activations.append('relu')
                    CONFIG['model']['activations'] = new_activations
            elif choice == '7':
                size = float(input("Enter validation size (e.g., 30 for 30%): "))
                CONFIG['split']['val_size'] = size / 100.0
            elif choice == '8':
                mode = input("Enter seed mode ('auto' or 'manual'): ").lower()
                if mode in ['auto', 'manual']:
                    CONFIG['split']['seed_mode'] = mode
                    if mode == 'manual':
                        CONFIG['split']['manual_seed'] = int(input("Enter seed value: "))
                else: print(f"{Colors.RED}Invalid mode.{Colors.NC}")
            elif choice == 'm': return
            else:
                print(f"{Colors.RED}Invalid choice.{Colors.NC}")
                time.sleep(1)
        except (ValueError, IndexError):
            print(f"{Colors.RED}Invalid input.{Colors.NC}")
            time.sleep(1)
            
def log_training_history(seed, history, log_file, final_accuracy):
    """Appends the detailed epoch-by-epoch history and final accuracy to a log file."""
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*20} Seed {seed} Training History {'='*20}\n")
        for i in range(len(history['train_loss'])):
            epoch = i + 1
            train_loss = history['train_loss'][i]
            train_acc = history['train_acc'][i]
            val_loss = history['val_loss'][i]
            val_acc = history['val_acc'][i]
            log_line = (f"Epoch {epoch:03d}/{len(history['train_loss']):03d} - "
                        f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                        f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}\n")
            f.write(log_line)
        f.write(f"--- Seed {seed} final validation accuracy: {final_accuracy*100:.2f}%\n")

def train_phase():
    print(f"\n{Colors.BLUE}--- Starting Training Process ---{Colors.NC}")
    X_full, y_full = load_data(CONFIG["dataset_path"])
    if X_full is None: return
    
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    TRAIN_PATH = 'temp_train_data.csv'
    VAL_PATH = 'temp_val_data.csv'
    
    print(f"{Colors.CYAN}Splitting data into a definitive Train/Validation set...{Colors.NC}")
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X_full))
    split_point = int(len(X_full) * (1 - CONFIG['split']['val_size']))
    train_idx, val_idx = shuffled_indices[:split_point], shuffled_indices[split_point:]
    X_train_raw, y_train_raw = X_full[train_idx], y_full[train_idx]
    X_val_raw, y_val_raw = X_full[val_idx], y_full[val_idx]

    preprocessor = Preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw).T
    y_train = y_train_raw.T
    X_val = preprocessor.transform(X_val_raw).T
    y_val = y_val_raw.T
    print("Data imputed and scaled correctly based on the definitive training set.")

    final_model, final_history = None, None

    # Auto seed mode
    if CONFIG['split']['seed_mode'] == 'auto':
        print(f"{Colors.CYAN}Seed mode is 'auto'. Searching for the best initialization seed...{Colors.NC}")
        best_seed, best_accuracy = -1, 0.0
        seed_search_results = []
        
        for seed in range(15):
            np.random.seed(seed)
            print(f"  - Training with initialization seed {seed}...")
            
            model = MLP([X_train.shape[0]] + CONFIG['model']['architecture'] + [2], CONFIG['model']['activations'])
            history = model.fit(X_train, y_train, X_val, y_val, epochs=CONFIG['model']['epochs'], 
                                learning_rate=CONFIG['model']['lr'], batch_size=CONFIG['model']['batch_size'],
                                patience=CONFIG['model']['patience'], verbose=False)
            
            _, val_acc = model.evaluate(X_val, y_val)
            log_training_history(seed, history, LOG_PATH, val_acc)
            seed_search_results.append({'seed': seed, 'accuracy': val_acc, 'history': history})
            print(f"    - Seed {seed} final validation accuracy: {Colors.YELLOW}{val_acc*100:.2f}%{Colors.NC}")

            if val_acc > best_accuracy:
                best_accuracy, best_seed = val_acc, seed
                final_model = model
                final_history = history
        
        print(f"\n{Colors.GREEN}Best initialization seed found: {best_seed} with validation accuracy {best_accuracy*100:.2f}%{Colors.NC}")
        print(f"Detailed training logs for all seeds have been saved to {Colors.GREEN}{LOG_PATH}{Colors.NC}")
        plot_seed_search(seed_search_results)

    else:
        # Manual seed mode
        final_seed = CONFIG['split']['manual_seed']
        print(f"\n{Colors.CYAN}Manual seed mode. Training with initialization seed {final_seed}.{Colors.NC}")
        np.random.seed(final_seed)
        layer_dims = [X_train.shape[0]] + CONFIG['model']['architecture'] + [2]
        final_model = MLP(layer_dims, CONFIG['model']['activations'])
        final_history = final_model.fit(X_train, y_train, X_val, y_val, epochs=CONFIG['model']['epochs'], 
                                      learning_rate=CONFIG['model']['lr'], batch_size=CONFIG['model']['batch_size'],
                                      patience=CONFIG['model']['patience'], verbose=True)

    if final_model:
        if not os.path.exists('saved_models'): os.makedirs('saved_models')
        final_model.save_model(MODEL_PATH, preprocessor, final_history)
        print(f"\n{Colors.BOLD}Best model trained and saved to {Colors.GREEN}{MODEL_PATH}{Colors.NC}")
    else:
        print(f"{Colors.RED}No model was trained successfully.{Colors.NC}")
    
    cleanup_files([TRAIN_PATH, VAL_PATH])

def predict_phase():
    print(f"\n{Colors.BLUE}--- Prediction and Evaluation ---{Colors.NC}")
    if not os.path.exists(MODEL_PATH):
        print(f"{Colors.RED}Model file '{MODEL_PATH}' not found. Please train the model first.{Colors.NC}")
        return

    while True:
        test_path = input("Enter path to the dataset for prediction: ")
        if not os.path.exists(test_path):
            print(f"{Colors.RED}File not found at '{test_path}'. Please try again.{Colors.NC}")
            continue
            
        # Validate that it's a proper CSV file
        is_valid, error_msg, sample_count, feature_count = validate_csv_file(test_path)
        
        if not is_valid:
            print(f"{Colors.RED}Error: {error_msg}{Colors.NC}")
            continue
            
        print(f"{Colors.GREEN}✓ Valid CSV file detected with {sample_count} samples and {feature_count} features.{Colors.NC}")
        break

    model, preprocessor, history = MLP.load_model(MODEL_PATH)
    
    X_raw, y_true_one_hot = load_data(test_path)
    if X_raw is None: return

    X = preprocessor.transform(X_raw).T
    y_true_one_hot_T = y_true_one_hot.T

    y_pred_probs = model.predict(X)

    y_n = y_true_one_hot_T[0, :]
    p_n = y_pred_probs[0, :]
    
    N = len(y_n)
    binary_cross_entropy = - (1/N) * np.sum(y_n * np.log(p_n + 1e-9) + (1 - y_n) * np.log(1 - p_n + 1e-9))

    print(f"\n{Colors.CYAN}--- Evaluation Result ---{Colors.NC}")
    print(f"Binary Cross-Entropy Error on '{os.path.basename(test_path)}': {Colors.BOLD}{Colors.GREEN}{binary_cross_entropy:.6f}{Colors.NC}")

    y_true_labels = np.argmax(y_true_one_hot_T, axis=0)
    y_pred_labels = np.argmax(y_pred_probs, axis=0)
    
    plot_prediction_results(history, y_true_labels, y_pred_labels, y_pred_probs)
    
    cleanup_files([PRED_PATH])
    
def view_history_phase():
    """Parses and displays the training history from the log file."""
    clear_screen()
    print(f"\n{Colors.BLUE}--- Training History Viewer ---{Colors.NC}")
    if not os.path.exists(LOG_PATH):
        print(f"{Colors.RED}Log file '{LOG_PATH}' not found. Please train a model first.{Colors.NC}")
        return

    with open(LOG_PATH, 'r') as f:
        content = f.read()

    seed_results = re.findall(r"--- Seed (\d+) final validation accuracy: ([\d.]+)%", content)
    if not seed_results:
        print(f"{Colors.YELLOW}No training history found in the log file.{Colors.NC}")
        return

    results_dict = {seed: float(acc) for seed, acc in seed_results}
    best_seed_str = max(results_dict, key=results_dict.get)
    
    other_seeds = sorted([s for s in results_dict.keys() if s != best_seed_str], key=int)

    while True:
        print("\nSelect a seed to view its detailed training log:")
        print(f"{Colors.GREEN}--- Best Seed ---{Colors.NC}")
        print(f"  - '{Colors.YELLOW}{best_seed_str}{Colors.NC}': Seed {best_seed_str} ({results_dict[best_seed_str]:.2f}% accuracy)")
        
        if other_seeds:
            print(f"\n{Colors.CYAN}--- Other Seeds ---{Colors.NC}")
            for seed in other_seeds:
                print(f"  - '{Colors.YELLOW}{seed}{Colors.NC}': Seed {seed} ({results_dict[seed]:.2f}% accuracy)")
        
        print(f"\n  - Enter '{Colors.YELLOW}all{Colors.NC}' to view all histories")
        print(f"  - Enter '{Colors.YELLOW}m{Colors.NC}' to return to the main menu")
        
        choice = input("Enter your choice: ").lower()

        if choice == 'm':
            return
        
        if choice == 'all':
            clear_screen()
            print(content)
            break
        elif choice in results_dict:
            clear_screen()
            history_block = re.search(f"====== Seed {choice} Training History ======(.*?)(?=\n--- Seed|$)", content, re.DOTALL)
            if history_block:
                print(f"{Colors.BOLD}{Colors.CYAN}====== Seed {choice} Training History ======{Colors.NC}")
                for line in history_block.group(1).strip().split('\n'):
                    colored_line = re.sub(r"(Epoch \d+/\d+)", f"{Colors.BOLD}\\1{Colors.NC}", line)
                    colored_line = re.sub(r"(loss: )([\d.]+)", f"\\1{Colors.GREEN}\\2{Colors.NC}", colored_line)
                    colored_line = re.sub(r"(acc: )([\d.]+)", f"\\1{Colors.GREEN}\\2{Colors.NC}", colored_line)
                    colored_line = re.sub(r"(val_loss: )([\d.]+)", f"\\1{Colors.YELLOW}\\2{Colors.NC}", colored_line)
                    colored_line = re.sub(r"(val_acc: )([\d.]+)", f"\\1{Colors.YELLOW}\\2{Colors.NC}", colored_line)
                    print(colored_line)
            break
        else:
            print(f"{Colors.RED}Invalid seed number. Please try again.{Colors.NC}")
            time.sleep(1)

def visualize_menu():
    while True:
        clear_screen()
        print(f"{Colors.BOLD}Visualize Menu{Colors.NC}")
        print(f"d - {Colors.CYAN}Describe dataset{Colors.NC}")
        print(f"p - {Colors.CYAN}Draw pair plot of key features{Colors.NC}")
        print(f"c - {Colors.CYAN}Draw correlation heatmap{Colors.NC}")
        print(f"m - {Colors.RED}Back to main menu{Colors.NC}")
        choice = input("Enter your choice: ").lower()

        if choice == 'd':
            clear_screen()
            describe(CONFIG["dataset_path"])
            input("\nPress Enter to continue...")
        elif choice == 'p':
            plot_pair(CONFIG["dataset_path"])
        elif choice == 'c':
            plot_correlation(CONFIG["dataset_path"])
        elif choice == 'm':
            return
        else:
            print(f"{Colors.RED}Invalid choice.{Colors.NC}")
            time.sleep(1)

def main_menu():
    """The main interactive menu of the application."""
    while True:
        clear_screen()
        print(f"{Colors.BOLD}Multilayer Perceptron Project{Colors.NC}")
        print(f"Current Training Dataset: {Colors.YELLOW}{CONFIG['dataset_path']}{Colors.NC}")
        print("-" * 40)
        print(f"c - {Colors.YELLOW}Configure Parameters{Colors.NC}")
        print(f"t - {Colors.GREEN}Train the model{Colors.NC}")
        
        if os.path.exists(LOG_PATH):
            print(f"h - {Colors.GREEN}See training history{Colors.NC}")

        print(f"p - {Colors.BLUE}Predict and evaluate{Colors.NC}")
        print(f"v - {Colors.MAGENTA}Visualize dataset{Colors.NC}")
        print(f"q - {Colors.RED}Quit{Colors.NC}")
        choice = input("Enter your choice: ").lower()

        pause = False
        if choice == 'c':
            configure_parameters_menu()
        elif choice == 't':
            train_phase()
            pause = True
        elif choice == 'h':
            if os.path.exists(LOG_PATH):
                view_history_phase()
                pause = True
            else:
                print(f"{Colors.RED}Invalid choice.{Colors.NC}")
                time.sleep(1)
        elif choice == 'p':
            predict_phase()
            pause = True
        elif choice == 'v':
            visualize_menu()
        elif choice == 'q':
            print("Exiting.")
            sys.exit(0)
        else:
            print(f"{Colors.RED}Invalid choice.{Colors.NC}")
            time.sleep(1)
        
        if pause:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    get_dataset_path()
    main_menu()