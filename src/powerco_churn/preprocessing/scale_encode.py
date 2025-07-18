import pandas as pd
from powerco_churn.utils.logger_utils import configure_logging
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import set_config

configure_logging(log_file_name = "scale_encode.log")

class ScaleEncode(BaseEstimator, TransformerMixin):
    
    def __init__(self):

        self.scaler = StandardScaler()
        self.onehotencoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
        set_config(transform_output = "pandas")

    def fit(self, X, y = None):

        self.numerical_features_ = [feature for feature in X.columns if pd.api.types.is_numeric_dtype(X[feature])
                                   and X[feature].nunique() > 2] 
        self.categorical_features_ = [feature for feature in X.columns if  pd.api.types.is_object_dtype(X[feature]) 
                                                             or isinstance(X[feature], pd.CategoricalDtype)] 
        X_categorical = X[self.categorical_features_].copy()
        self.scaler.fit(X[self.numerical_features_])
        self.onehotencoder.fit(X_categorical)
        self.columns = self.onehotencoder.get_feature_names_out(X_categorical.columns)

        return self

    def transform(self, X):

        check_is_fitted(self, ['numerical_features_', 'categorical_features_'])
        X_copy = X.copy()
        X_copy[self.numerical_features_] = self.scaler.transform(X_copy[self.numerical_features_])
        if not isinstance(X_copy, pd.DataFrame):
            X_copy = pd.DataFrame(X_copy, index = X.index, columns = X.columns)
        X_copy_encoded = self.onehotencoder.transform(X_copy[self.categorical_features_])
        if not isinstance(X_copy_encoded, pd.DataFrame):
            X_copy_encoded = pd.DataFrame(X_copy_encoded, index = X.index, columns = self.columns)
        X_copy = X_copy.drop(columns = self.categorical_features_)
        X_copy = pd.concat([X_copy, X_copy_encoded], axis = 1)

        return X_copy

if __name__ == "__main__":
     
    current_file = Path(__file__).resolve()
    base_path = current_file.parents[3]  # 0 is file, 1 is parent, ..., 3 = three levels up
    base_path = str(base_path)
    client_data_train = pd.read_csv(base_path + '/data/raw/train/train_client_data_raw.csv')
    price_data_train = pd.read_csv(base_path + '/data/raw/train/train_price_data_raw.csv')
    
    scaler_encoder = ScaleEncode()
    client_data_cleaned = scaler_encoder.fit_transform(client_data_train)
