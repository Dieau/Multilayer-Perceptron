import pandas as pd
import numpy as np
from mlp.utils import Colors

def describe(filepath):
    """
    Reads a dataset and prints a well-formatted, grouped statistical description.
    """
    print(f"{Colors.BLUE}--- Describing Dataset: {filepath} ---{Colors.NC}")
    try:
        # 30 features in the WBCD dataset
        feature_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
            'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
            'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
            'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
            'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        
        df = pd.read_csv(filepath, header=None)
        
        if df.shape[1] == 32: # Original data.csv with ID
            numeric_df = df.iloc[:, 2:]
        elif df.shape[1] == 31: # Split file without ID
            numeric_df = df.iloc[:, 1:]
        else:
            print(f"{Colors.YELLOW}Unexpected number of columns. Describing all numeric columns.{Colors.NC}")
            numeric_df = df.select_dtypes(include=np.number)

        numeric_df.columns = feature_names[:numeric_df.shape[1]]

        stats = {
            'Count': numeric_df.count(), 'Mean': numeric_df.mean(), 'Std': numeric_df.std(),
            'Min': numeric_df.min(), '25%': numeric_df.quantile(0.25),
            '50%': numeric_df.quantile(0.50), '75%': numeric_df.quantile(0.75),
            'Max': numeric_df.max()
        }
        description = pd.DataFrame(stats)
        
        groups = {
            "Mean Features": [name for name in feature_names if 'mean' in name],
            "Standard Error (SE) Features": [name for name in feature_names if 'se' in name],
            "Worst (Largest) Features": [name for name in feature_names if 'worst' in name]
        }

        for group_title, group_features in groups.items():
            print(f"\n{Colors.CYAN}--- {group_title} ---{Colors.NC}")
            
            group_df = description.loc[group_features]

            # Header
            header = f"{'Feature':<28}" + "".join([f"{stat:^15}" for stat in group_df.columns])
            print(f"{Colors.BOLD}{header}{Colors.NC}")
            print(f"{'-' * len(header)}")

            # Rows
            for feature, row in group_df.iterrows():
                row_str = f"{feature:<28}" + "".join([f"{val:^15.6f}" for val in row])
                print(row_str)

    except FileNotFoundError:
        print(f"{Colors.RED}Error: Dataset file not found at '{filepath}'{Colors.NC}")
    except Exception as e:
        print(f"{Colors.RED}An error occurred: {e}{Colors.NC}")