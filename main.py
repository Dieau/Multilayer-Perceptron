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

# --- Global Configuration State with UPDATED DEFAULTS ---
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

def get_dataset_path():
    while True:
        clear_screen()
        path = input("Please enter the path to the dataset CSV file to be used for training: ")
        if os.path.exists(path):
            CONFIG["dataset_path"] = path
            return
        else:
            print(f"{Colors.RED}File not found at '{path}'. Please try again.{Colors.NC}")
            input("Press Enter to continue...")

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
        print("m. Back to Main Menu")
        
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
                    input("Press Enter to continue...")
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
                input("Press Enter to continue...")
        except (ValueError, IndexError):
            print(f"{Colors.RED}Invalid input.{Colors.NC}")
            input("Press Enter to continue...")

def train_phase():
    print(f"\n{Colors.BLUE}--- Starting Training Process ---{Colors.NC}")
    X_full, y_full = load_data(CONFIG["dataset_path"])
    if X_full is None: return
    
    # Create temporary split files for this session
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

    if CONFIG['split']['seed_mode'] == 'auto':
        print(f"{Colors.CYAN}Seed mode is 'auto'. Searching for the best initialization seed...{Colors.NC}")
        best_seed, best_accuracy = -1, 0.0
        seed_search_results = []
        
        for seed in range(10):
            np.random.seed(seed)
            print(f"  - Training with initialization seed {seed}...")
            
            model = MLP([X_train.shape[0]] + CONFIG['model']['architecture'] + [2], CONFIG['model']['activations'])
            history = model.fit(X_train, y_train, X_val, y_val, epochs=CONFIG['model']['epochs'], 
                                learning_rate=CONFIG['model']['lr'], batch_size=CONFIG['model']['batch_size'],
                                patience=CONFIG['model']['patience'], verbose=True)
            
            _, val_acc = model.evaluate(X_val, y_val)
            seed_search_results.append({'seed': seed, 'accuracy': val_acc, 'history': history})
            print(f"    - Seed {seed} final validation accuracy: {Colors.YELLOW}{val_acc*100:.2f}%{Colors.NC}")

            if val_acc > best_accuracy:
                best_accuracy, best_seed = val_acc, seed
                final_model = model
                final_history = history
        
        print(f"\n{Colors.GREEN}Best initialization seed found: {best_seed} with validation accuracy {best_accuracy*100:.2f}%{Colors.NC}")
        plot_seed_search(seed_search_results)
    else:
        final_seed = CONFIG['split']['manual_seed']
        print(f"\n{Colors.CYAN}Manual seed mode. Training with initialization seed {final_seed}.{Colors.NC}")
        np.random.seed(final_seed)
        layer_dims = [X_train.shape[0]] + CONFIG['model']['architecture'] + [2]
        final_model = MLP(layer_dims, CONFIG['model']['activations'])
        final_history = final_model.fit(X_train, y_train, X_val, y_val, epochs=CONFIG['model']['epochs'], 
                                      learning_rate=CONFIG['model']['lr'], batch_size=CONFIG['model']['batch_size'],
                                      patience=CONFIG['model']['patience'])

    if final_model:
        if not os.path.exists('saved_models'): os.makedirs('saved_models')
        final_model.save_model(MODEL_PATH, preprocessor, final_history)
        print(f"\n{Colors.BOLD}Best model trained and saved to {Colors.GREEN}{MODEL_PATH}{Colors.NC}")
    else:
        print(f"{Colors.RED}No model was trained successfully.{Colors.NC}")
    
    # Cleanup temporary files
    cleanup_files([TRAIN_PATH, VAL_PATH])

def predict_phase():
    print(f"\n{Colors.BLUE}--- Prediction and Evaluation ---{Colors.NC}")
    if not os.path.exists(MODEL_PATH):
        print(f"{Colors.RED}Model file '{MODEL_PATH}' not found. Please train the model first.{Colors.NC}")
        return

    # FIX: Loop until a valid file path is provided
    while True:
        test_path = input("Enter path to the dataset for prediction: ")
        if os.path.exists(test_path):
            break
        else:
            print(f"{Colors.RED}File not found. Aborting.{Colors.NC}")
            # Immediately ask again
            print(f"{Colors.RED}File not found at '{test_path}'. Please try again.{Colors.NC}")

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

def visualize_menu():
    while True:
        clear_screen()
        print(f"{Colors.BOLD}Visualize Menu{Colors.NC}")
        print("d - Describe dataset")
        print("p - Draw pair plot of key features")
        print("c - Draw correlation heatmap")
        print("m - Back to main menu")
        choice = input("Enter your choice: ").lower()

        if choice == 'd':
            clear_screen()
            describe(CONFIG["dataset_path"])
            input("\nPress Enter to continue...")
        elif choice == 'p':
            plot_pair(CONFIG["dataset_path"])
            input("\nPress Enter to continue...")
        elif choice == 'c':
            plot_correlation(CONFIG["dataset_path"])
            input("\nPress Enter to continue...")
        elif choice == 'm':
            return # Return to main menu without pausing
        else:
            print(f"{Colors.RED}Invalid choice.{Colors.NC}")

def main_menu():
    """The main interactive menu of the application."""
    while True:
        clear_screen()
        print(f"{Colors.BOLD}Multilayer Perceptron Project{Colors.NC}")
        print(f"Current Training Dataset: {Colors.YELLOW}{CONFIG['dataset_path']}{Colors.NC}")
        print("-" * 40)
        print("c - Configure Parameters")
        print("t - Train the model")
        print("p - Predict and evaluate")
        print("v - Visualize dataset")
        print("q - Quit")
        choice = input("Enter your choice: ").lower()

        # FIX: Remove the blanket "Press Enter" and handle pausing on a case-by-case basis
        if choice == 'c':
            configure_parameters_menu()
        elif choice == 't':
            train_phase()
            input("\nPress Enter to continue...")
        elif choice == 'p':
            predict_phase()
            input("\nPress Enter to continue...")
        elif choice == 'v':
            visualize_menu()
        elif choice == 'q':
            print("Exiting.")
            sys.exit(0)
        else:
            print(f"{Colors.RED}Invalid choice.{Colors.NC}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    get_dataset_path()
    main_menu()