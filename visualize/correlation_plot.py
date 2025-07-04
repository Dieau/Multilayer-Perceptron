import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlp.utils import Colors

def plot_correlation(filepath):
    """
    Calculates and plots a correlation heatmap for the dataset's numerical features.
    """
    print(f"{Colors.BLUE}--- Generating Correlation Heatmap for: {filepath} ---{Colors.NC}")
    try:
        df = pd.read_csv(filepath, header=None)
        numeric_df = df.select_dtypes(include=np.number)
        
        if numeric_df.shape[1] < 2:
            print(f"{Colors.YELLOW}Not enough numerical features for a correlation plot.{Colors.NC}")
            return

        numeric_df.columns = [f'F{i}' for i in range(numeric_df.shape[1])]
        
        print("Calculating correlation matrix...")
        corr_matrix = numeric_df.corr()

        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='viridis', linewidths=.5)
        plt.title('Feature Correlation Heatmap', fontsize=16)
        plt.show()

    except FileNotFoundError:
        print(f"{Colors.RED}Error: Dataset file not found at '{filepath}'{Colors.NC}")
    except Exception as e:
        print(f"{Colors.RED}An error occurred: {e}{Colors.NC}")