from powerco_churn.preprocessing.date_processing import DateParserTransformer
from powerco_churn.preprocessing.date_processing import CreatingDateFeatures
from powerco_churn.preprocessing.replace_outliers import ReplaceOutliers
from powerco_churn.preprocessing.reduce_skew import ReduceSkew
from powerco_churn.preprocessing.dropping_missing import DropMissing
from powerco_churn.preprocessing.scale_encode import ScaleEncode


from powerco_churn.preprocessing.temporal_series_processing import TransformPricesTemporalSeries
from powerco_churn.EDA.basic_data_wrangling import basic_wrangling
from powerco_churn.utils.log_mlflow_utils import log_dataset_mlflow

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

import sys
import os

import mlflow.sklearn
import mlflow



pipeline_client_data = Pipeline(
    [('parse_dates', DateParserTransformer(date_columns = ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal'])),
     ('remove_missing', DropMissing()),
     ('create_date_features', CreatingDateFeatures(date_columns = ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal']))]
)

pipeline_price_series = Pipeline(
    [('remove_missing', DropMissing()),
     ('transform_prices', TransformPricesTemporalSeries(price_columns = ['price_off_peak_var',
                                                                         'price_peak_var',
                                                                         'price_mid_peak_var',
                                                                         'price_off_peak_fix',
                                                                         'price_peak_fix',
                                                                         'price_mid_peak_fix']))])

class MultiInputMerger(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipeline_client_data = pipeline_client_data
        self.pipeline_price_series = pipeline_price_series

    def fit(self, X, y=None):
        df1, df2 = X
        self.pipeline_client_data.fit(df1)
        self.pipeline_price_series.fit(df2)
        return self

    def transform(self, X):
        df1, df2 = X
        columns_df1 = df1.columns
        columns_df2 = df2.columns
        df1_preprocessed = self.pipeline_client_data.transform(df1)
        df2_preprocessed = self.pipeline_price_series.transform(df2)

        # Convert to DataFrame if necessary
        if not isinstance(df1_preprocessed, pd.DataFrame):
            df1_preprocessed = pd.DataFrame(df1_preprocessed, 
                                            index = df1.index,
                                            columns = columns_df1)
        if not isinstance(df2_preprocessed, pd.DataFrame):
            df2_preprocessed = pd.DataFrame(df2_preprocessed, 
                                            index = df2.index,
                                            columns = columns_df2)
            
        merged = pd.merge(df1_preprocessed,
                        df2_preprocessed,
                        on = "id",
                        how = "inner")

        return merged




def get_preprocess_pipeline():

    pipeline = Pipeline([
    ('merge', MultiInputMerger()),
    ('basic wrangling', FunctionTransformer(basic_wrangling)),
    ('replacing outliers',ReplaceOutliers()),
    ('reduce skew', ReduceSkew()),
    ('scale and encode', ScaleEncode())
     ])
    return pipeline


if __name__ == '__main__':

    current_file = Path(__file__).resolve()
    base_path = current_file.parents[3]  # 0 is file, 1 is parent, ..., 3 = three levels up
    base_path = str(base_path)

    client_data_train = pd.read_csv(base_path + '/data/raw/train/train_client_data_raw.csv')
    price_data_train = pd.read_csv(base_path + '/data/raw/train/train_price_data_raw.csv')

    client_data_test = pd.read_csv(base_path + '/data/raw/test/test_client_data_raw.csv')
    price_data_test = pd.read_csv(base_path + '/data/raw/test/test_price_data_raw.csv')
    
    y_train = pd.read_csv(base_path + '/data/raw/train/y_train.csv')
    y_test = pd.read_csv(base_path + '/data/raw/test/y_test.csv')

    mlflow.set_tracking_uri(f"file:{base_path}/mlruns")  # Goes one level up
    mlflow.set_experiment("powerco_churn")

    pipeline = get_preprocess_pipeline()
    x_train = pipeline.fit_transform([client_data_train, price_data_train])
    x_test = pipeline.transform([client_data_test, price_data_test])

    x_train.to_csv(base_path + '/data/cleaned/train/x_train.csv', index = False)
    x_test.to_csv(base_path + '/data/cleaned/test/x_test.csv', index = False)

    y_train.to_csv(base_path + '/data/cleaned/train/y_train.csv', index = False)
    y_test.to_csv(base_path + '/data/cleaned/test/y_test.csv', index = False)


    with mlflow.start_run(run_name = 'Preprocessing Pipeline'):
        mlflow.sklearn.log_model(
        sk_model = pipeline,
        name = "preprocessing_pipeline",
        registered_model_name=None  # Optional: register in Model Registry
    )
        log_dataset_mlflow(x_train, source = base_path + '/data/cleaned/train/x_train.csv', name = 'x_train', context = 'training') 
        log_dataset_mlflow(x_train, source = base_path + '/data/cleaned/train/x_test.csv', name = 'x_test', context = 'testing')

        log_dataset_mlflow(y_train, source = base_path + '/data/cleaned/train/y_train.csv', name = 'y_train', context = 'training') 
        log_dataset_mlflow(y_train, source = base_path + '/data/cleaned/train/y_test.csv', name = 'y_test', context = 'testing')


    