import pandas as pd
from powerco_churn.utils.logger_utils import configure_logging
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import PowerTransformer
from powerco_churn.EDA.skewness import correct_skew
import numpy as np


class ReduceSkew(BaseEstimator, TransformerMixin):
    
    def __init__(self, skew_threshold=0.5):
        self.skew_threshold = skew_threshold
        self.best_transformation_ = {}
        self.min_max_values_ = defaultdict(dict)
        self.yeo_features = []
        self._yeo_transformer = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        X_copy = X.copy()

        for feature in X_copy.columns:
            if pd.api.types.is_numeric_dtype(X_copy[feature]):
                skew = X_copy[feature].skew()

                if abs(skew) > self.skew_threshold:
                    _, _, transform_type = correct_skew(X_copy, feature)

                    if transform_type != 'Converted to binary' and transform_type != 'yeo':
                        if X_copy[feature].min() < 0:
                            transform_type = 'yeo'

                    self.best_transformation_[feature] = transform_type

                    if transform_type == 'Converted to binary':
                        self.min_max_values_[feature] = {
                            'min_value': X_copy[feature].min(),
                            'max_value': X_copy[feature].max()
                        }

                    if transform_type == 'yeo':
                        self.yeo_features.append(feature)

        if self.yeo_features:
            self._yeo_transformer.fit(X_copy[self.yeo_features])

        return self

    def _apply_power_transformer(self, X, feature, power):
        power = float(power)
        X[feature] = np.power(X[feature], power)
        return X

    def _convert_binary(self, X, feature):
        min_val = self.min_max_values_[feature]['min_value']
        max_val = self.min_max_values_[feature]['max_value']
        skew = X[feature].skew()

        if skew > 0:
            X[feature] = (X[feature] != min_val).astype(int)
        else:
            X[feature] = (X[feature] != max_val).astype(int)

        return X

    def transform(self, X):
        X_copy = X.copy()

        for feature, transform_type in self.best_transformation_.items():
            if transform_type == 'Converted to binary':
                X_copy = self._convert_binary(X_copy, feature)

            elif transform_type not in ['Converted to binary', 'yeo']:
                X_copy = self._apply_power_transformer(X_copy, feature, transform_type)

        if self.yeo_features:
            X_copy[self.yeo_features] = self._yeo_transformer.transform(X_copy[self.yeo_features])

        return X_copy
    
if __name__ == "__main__":

    configure_logging(log_file_name = "reduce_skew.log") 
    current_file = Path(__file__).resolve()
    base_path = current_file.parents[3]  # 0 is file, 1 is parent, ..., 3 = three levels up
    base_path = str(base_path)
    client_data_train = pd.read_csv(base_path + '/data/raw/train/train_client_data_raw.csv')
    price_data_train = pd.read_csv(base_path + '/data/raw/train/train_price_data_raw.csv')
    
    skew_reducer = ReduceSkew()
    client_data_cleaned = skew_reducer.fit_transform(client_data_train)
