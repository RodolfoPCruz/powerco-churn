import pandas as pd
import logging
from powerco_churn.utils.logger_utils import configure_logging
from sklearn.base import BaseEstimator, TransformerMixin
from powerco_churn.EDA.outliers import calculate_outlier_threshold
from pathlib import Path


class ReplaceOutliers(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_outliers_dict_ = {}

        
    def fit(self, X, y = None):
        for feature in X.columns:
            if pd.api.types.is_numeric_dtype(X[feature]):
                min_thresh, max_thresh = calculate_outlier_threshold(X, feature)
                self.feature_outliers_dict_[feature] = {'max_thresh': max_thresh, 'min_thresh': min_thresh}
        return self

    def transform(self, X):
        X_copy = X.copy()
        logging.info("Replacing outliers...")
        for feature in self.feature_outliers_dict_:
            num_outliers = (X_copy[feature] > self.feature_outliers_dict_[feature]['max_thresh']).sum()
            num_outliers += (X_copy[feature] < self.feature_outliers_dict_[feature]['min_thresh']).sum()
            logging.info(f'Number of outliers in {feature}: {num_outliers}') 

            X_copy[feature] = X_copy[feature].mask(X_copy[feature] > self.feature_outliers_dict_[feature]['max_thresh'], 
                                        self.feature_outliers_dict_[feature]['max_thresh'])
            X_copy[feature] = X_copy[feature].mask(X_copy[feature] < self.feature_outliers_dict_[feature]['min_thresh'], 
                                        self.feature_outliers_dict_[feature]['min_thresh'])
        
        return X_copy

if __name__ == "__main__":
    
    configure_logging(log_file_name = "replace_outliers.log")
    current_file = Path(__file__).resolve()
    base_path = current_file.parents[3]  # 0 is file, 1 is parent, ..., 3 = three levels up
    base_path = str(base_path)
    client_data_train = pd.read_csv(base_path + '/data/raw/train/train_client_data_raw.csv')
    price_data_train = pd.read_csv(base_path + '/data/raw/train/train_price_data_raw.csv')
    
    outliers_replacer = ReplaceOutliers()
    client_data_cleaned = outliers_replacer.fit_transform(client_data_train)
