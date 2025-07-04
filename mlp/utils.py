import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Colors:
    """ANSI color codes for formatted terminal output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    NC = '\033[0m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'

class Scaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return (data - self.min) / (self.max - self.min + 1e-8)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

def load_data(filepath):
    """
    Loads data from the specified file. It no longer converts to float,
    allowing for missing values to be handled by the Preprocessor.
    """
    try:
        df = pd.read_csv(filepath, header=None)
        
        if df.shape[1] == 32:
            df = df.drop(columns=[0])
        
        # Replace empty strings or placeholders with NaN for consistent handling
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        y_categorical = df.iloc[:, 0]
        X = df.iloc[:, 1:].values
        
        y = np.zeros((y_categorical.shape[0], 2))
        y[y_categorical == 'M', 0] = 1
        y[y_categorical == 'B', 1] = 1
        return X, y

    except FileNotFoundError:
        print(f"{Colors.RED}Error: Data file not found at '{filepath}'{Colors.NC}")
        return None, None
    except Exception as e:
        print(f"{Colors.RED}Error processing data file '{filepath}': {e}{Colors.NC}")
        return None, None

def plot_learning_curves(history, title_suffix=''):
    """Plots loss and accuracy curves for training and validation sets."""
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curves {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    if 'val_acc' in history and history['val_acc']:
        plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title(f'Accuracy Curves {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.suptitle(f'Model Training Performance {title_suffix}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])