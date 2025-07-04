import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlp.utils import Colors

def plot_pair(filepath):
    """
    Plots a pair plot for a subset of key features from the dataset.
    """
    print(f"{Colors.BLUE}--- Generating Pair Plot for: {filepath} ---{Colors.NC}")
    try:
        df = pd.read_csv(filepath, header=None)
        
        column_names = [
            'ID', 'Diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 
            'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean'
        ]
        
        subset_cols_indices = [1, 2, 3, 6, 7]
        subset_cols_names = [column_names[i] for i in subset_cols_indices]
        
        plot_df = df[subset_cols_indices]
        plot_df.columns = subset_cols_names

        print("Generating plot... This may take a moment.")
        sns.pairplot(plot_df, hue='Diagnosis', palette={'M': 'red', 'B': 'blue'})
        plt.suptitle('Pair Plot of Key Features', y=1.02)
        plt.show()

    except FileNotFoundError:
        print(f"{Colors.RED}Error: Dataset file not found at '{filepath}'{Colors.NC}")
    except Exception as e:
        print(f"{Colors.RED}An error occurred: {e}{Colors.NC}")