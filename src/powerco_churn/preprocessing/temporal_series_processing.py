import logging
import pandas as pd
from powerco_churn.utils.logger_utils import configure_logging
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
from scipy.stats import linregress
import numpy as np
from functools import reduce


class TransformPricesTemporalSeries(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                price_columns,
                id_column = 'id',
                date_column = 'price_date',
                verbose = True):
        self.price_columns = price_columns
        self.id_column = id_column
        self.date_column = date_column
        self.verbose = verbose

    def fit(self, X, y = None):
        return self

    def _calculating_mean_prices(self, X):
        if self.verbose:
            logging.info("Calculating mean prices...")

        X_mean = X.groupby(self.id_column)[self.price_columns].mean().reset_index()
        X_mean_columns = [self.id_column] + [f'mean_{col}' for col in X_mean.columns[1:]]
        X_mean.columns = X_mean_columns

        energy_features = [feature for feature in X_mean_columns if 'var' in feature]
        potency_features = [feature for feature in X_mean_columns if 'fix' in feature]

        X_mean['mean_energy_price'] = X_mean[energy_features].mean(axis = 1)
        X_mean['mean_potency_price'] = X_mean[potency_features].mean(axis = 1)

        X_mean['energy_peak_minus_offpeak']  = X_mean['mean_price_peak_var'] - X_mean['mean_price_off_peak_var']
        X_mean['potency_peak_minus_offpeak'] = X_mean['mean_price_peak_fix'] - X_mean['mean_price_off_peak_fix']

        return X_mean

    def _last_prices(self, X):

        if self.verbose:
            logging.info("Getting last prices...")

        last_price = X.loc[X.groupby(self.id_column)[self.date_column].idxmax()]
        last_price = last_price.reset_index(drop=True)
        last_price = last_price.drop(columns = [self.date_column])
        last_price.columns = ['id'] + [f'last_{col}' for col in last_price.columns[1:]] 
        
        return last_price

    def _difference_last_first_prices(self, X):

        if self.verbose:
            logging.info("Calculating the differece between last and first prices...")


        last = X.loc[X.groupby(self.id_column)[self.date_column].idxmax()].set_index(self.id_column, drop = True)
        last = last.drop(columns = [self.date_column])
        first = X.loc[X.groupby(self.id_column)[self.date_column].idxmin()].set_index(self.id_column, drop = True)
        first = first.drop(columns = [self.date_column])

        difference = last - first
        difference = difference.reset_index(drop = False)
        difference.columns = ['id'] + [f'difference_{col}' for col in difference.columns[1:]]

        return difference


    def _calculating_slopes_linear_regression(self, X):

        if self.verbose:
            logging.info("Calculating the slopes of the linear regression...")

        agg_dict = {
        f'{col}_slope' : (col, lambda x: linregress(np.arange(len(x)), x).slope)
        for col in self.price_columns
        }

        slopes_regression = X.groupby(self.id_column).agg(**agg_dict)
        slopes_regression = slopes_regression.reset_index()
        slopes_regression.head()

        return slopes_regression

    def _calculating_variance(self, X):

        if self.verbose:
            logging.info("Calculating the standard deviation of prices...")

        std_prices = X.groupby(self.id_column)[self.price_columns].std()
        std_prices = std_prices.reset_index()
        std_columns = ['id'] + [f'std_{col}' for col in std_prices.columns[1:]]
        std_prices.columns = std_columns

        return std_prices


    def transform(self, X):
        X_copy = X.copy()

        
        X_mean = self._calculating_mean_prices(X_copy)
        X_last = self._last_prices(X_copy)
        X_difference = self._difference_last_first_prices(X_copy)
        X_slope = self._calculating_slopes_linear_regression(X_copy)
        X_std = self._calculating_variance(X_copy)

        df_list = [X_mean, X_last, X_difference, X_slope, X_std]
        
        prices_df = reduce(lambda left, right: pd.merge(left, right, on='id', how='inner'), df_list)

        return prices_df

if __name__ == "__main__":
    
    configure_logging(log_file_name = "temporal_series_processing.log")
    calculating_temporal_series = TransformPricesTemporalSeries(price_columns = ['price_off_peak_var',
                                                                        'price_peak_var',
                                                                        'price_mid_peak_var',
                                                                        'price_off_peak_fix',
                                                                        'price_peak_fix',
                                                                     'price_mid_peak_fix'])
    
    current_file = Path(__file__).resolve()
    base_path = current_file.parents[3]  # 0 is file, 1 is parent, ..., 3 = three levels up
    base_path = str(base_path)
    #client_data_train = pd.read_csv(base_path + '/data/raw/train/train_client_data_raw.csv')
    price_data_train = pd.read_csv(base_path + '/data/raw/train/train_price_data_raw.csv')
    prices_features = calculating_temporal_series.fit_transform(price_data_train)
    
    