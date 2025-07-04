import numpy as np
import pandas as pd
from .utils import Scaler

class Preprocessor:
    """
    Handles all data preprocessing steps: imputation and scaling.
    It learns parameters from the training set and applies them consistently.
    """
    def __init__(self):
        self.imputation_values = None
        self.scaler = Scaler()

    def _coerce_to_numeric(self, data):
        """Safely convert data to a numeric format, coercing errors to NaN."""
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        return data.values.astype(float)

    def fit(self, data):
        """
        Learns imputation and scaling parameters from the training data.
        """
        numeric_data = self._coerce_to_numeric(data)

        self.imputation_values = np.nanmean(numeric_data, axis=0)
        
        nan_indices = np.where(np.isnan(numeric_data))
        numeric_data[nan_indices] = np.take(self.imputation_values, nan_indices[1])
        
        self.scaler.fit(numeric_data)

    def transform(self, data):
        """
        Applies the learned imputation and scaling to new data.
        """
        if self.imputation_values is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")

        numeric_data = self._coerce_to_numeric(data)

        nan_indices = np.where(np.isnan(numeric_data))
        numeric_data[nan_indices] = np.take(self.imputation_values, nan_indices[1])

        return self.scaler.transform(numeric_data)

    def fit_transform(self, data):
        """A convenience method to fit and transform the data in one step."""
        self.fit(data)
        return self.transform(data)