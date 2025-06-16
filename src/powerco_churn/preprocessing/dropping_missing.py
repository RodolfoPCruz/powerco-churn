from powerco_churn.utils.logger_utils import configure_logging
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import logging

configure_logging(log_file_name = "drop_missing_values.log")

class DropMissing(BaseEstimator, TransformerMixin):
    def __init__(self, axis = 0, verbose = True):
        self.axis = axis
        self.verbose = verbose

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        if self.verbose:
            logging.info("Dropping missing values...")
        n_rows_initial, n_cols_initial = X.shape[0], X.shape[1]
        X.dropna(axis = self.axis)
        n_rows_final, n_cols_final = X.shape[0], X.shape[1]

        if n_rows_initial != n_rows_final and self.verbose:
            logging.info(f"Removed {n_rows_initial - n_rows_final} rows with missing values.")
        
        if n_cols_initial != n_cols_final and self.verbose:
            logging.info(f"Removed {n_cols_initial - n_cols_final} columns with missing values.")

        if n_rows_initial == n_rows_final and n_cols_initial == n_cols_final and self.verbose:
            logging.info("No missing values found.")

        return X 

if __name__ == "__main__":
    drop_missing = DropMissing()
    
    current_file = Path(__file__).resolve()
    base_path = current_file.parents[3]  # 0 is file, 1 is parent, ..., 3 = three levels up
    base_path = str(base_path)
    client_data_train = pd.read_csv(base_path + '/data/raw/train/train_client_data_raw.csv')
    price_data_train = pd.read_csv(base_path + '/data/raw/train/train_price_data_raw.csv')
    client_data_no_missing = drop_missing.fit_transform(client_data_train)
    
    