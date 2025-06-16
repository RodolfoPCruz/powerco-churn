from sklearn.base import BaseEstimator, TransformerMixin
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from powerco_churn.utils.logger_utils import configure_logging

configure_logging(log_file_name = "date_processing.log")

class DateParserTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_columns, standard_format = '%Y-%m-%d', verbose = True):
        self.date_columns = date_columns
        self.standard_format = standard_format
        self.verbose = verbose  

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.verbose:
            logging.info("Parsing and formatting dates...")

        def parse_and_format(date_string):
            formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d %b %Y", "%B %d, %Y"]
            if not isinstance(date_string, str):
                return pd.Nat
            for fmt in formats:
                try:
                    return datetime.strptime(date_string, fmt).strftime(self.standard_format)
                except ValueError:
                    continue
            return pd.NaT

        for col in self.date_columns:
            X_copy[col] = X_copy[col].apply(parse_and_format)
        return X_copy    

class CreatingDateFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                date_columns, 
                reference_date = '2020-01-01', 
                start_date_feature = 'date_activ', 
                final_date_feature = 'date_end', 
                renewal_date_feature = 'date_renewal',
                modification_date_feature = 'date_modif_prod',
                verbose = True,
                drop_original_date_features = True):

        self.date_columns = date_columns
        self.reference_date = pd.to_datetime(reference_date)
        self.start_date_feature = start_date_feature
        self.final_date_feature = final_date_feature
        self.modification_date_feature = modification_date_feature
        self.renewal_date_feature = renewal_date_feature
        self.drop_original_date_features = drop_original_date_features
        self.verbose = verbose

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        if self.verbose:
            logging.info("Creating date features...")
        # Ensure datetime conversion
        for col in [self.start_date_feature, 
                    self.final_date_feature, 
                    self.modification_date_feature,
                    self.renewal_date_feature]:
            X_copy[col] = pd.to_datetime(X_copy[col], errors='coerce')


        X_copy['contract_length'] = (X_copy[self.final_date_feature] - X_copy[self.start_date_feature]).dt.days
        X_copy['days_until_end'] = (self.reference_date - X_copy[self.final_date_feature]).dt.days
        X_copy['days_until_renewal'] = (self.reference_date - X_copy[self.renewal_date_feature]).dt.days
        X_copy['days_since_modification'] = (self.reference_date - X_copy[self.modification_date_feature]).dt.days

        if self.drop_original_date_features:
            X_copy = X_copy.drop(columns = self.date_columns)

        if 'contract_length' in X_copy.columns and self.verbose:
            logging.info("contract_length feature created.")

        if 'days_until_end' in X_copy.columns and self.verbose:
            logging.info("days_until_end feature created.")

        if 'days_until_renewal' in X_copy.columns and self.verbose:
            logging.info("days_until_renewal feature created.")

        if 'days_since_modification' in X_copy.columns and self.verbose:
            logging.info("days_since_modification feature created.")
        

        return X_copy

if __name__ == "__main__":
    date_parser = DateParserTransformer(date_columns = ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal'], verbose = False)
    feature_creator = CreatingDateFeatures(date_columns = ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal'], verbose = False)
    
    current_file = Path(__file__).resolve()
    base_path = current_file.parents[3]  # 0 is file, 1 is parent, ..., 3 = three levels up
    base_path = str(base_path)
    client_data_train = pd.read_csv(base_path + '/data/raw/train/train_client_data_raw.csv')
    price_data_train = pd.read_csv(base_path + '/data/raw/train/train_price_data_raw.csv')
    
    teste_parser = date_parser.fit_transform(client_data_train)
    teste_creating_features = feature_creator.fit_transform(teste_parser)
