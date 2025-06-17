from powerco_churn.preprocessing.date_processing import DateParserTransformer
from powerco_churn.preprocessing.date_processing import CreatingDateFeatures
from powerco_churn.preprocessing.replace_outliers import ReplaceOutliers
from powerco_churn.preprocessing.reduce_skew import ReduceSkew
from powerco_churn.preprocessing.dropping_missing import DropMissing
from powerco_churn.preprocessing.scale_encode import ScaleEncode


from powerco_churn.preprocessing.temporal_series_processing import TransformPricesTemporalSeries
from powerco_churn.EDA.basic_data_wrangling import basic_wrangling


from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import FunctionTransformer



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

def merge_data(X):

    df1, df2 = X
    df1_preprocessed = pipeline_client_data.fit_transform(df1)
    df2_preprocessed = pipeline_price_series.fit_transform(df2)

    #Convert to dataframe if necessary
    if not isinstance(df1_preprocessed, pd.DataFrame):
        df1_preprocessed = pd.DataFrame(df1_preprocessed, index = df1.index)

    if not isinstance(df2_preprocessed, pd.DataFrame):
        df2_preprocessed = pd.DataFrame(df2_preprocessed, index = df2.index)
    
    #logging.info("Merging dataframes...")
    merged = pd.merge(
        df1_preprocessed,
        df2_preprocessed,
        on='id',
        how='inner'
    )

    return merged



pipeline = Pipeline([
    ('merge', FunctionTransformer(merge_data, validate = False)),
    ('basic wrangling', FunctionTransformer(basic_wrangling)),
    ('replacing outliers',ReplaceOutliers()),
    ('reduce skew', ReduceSkew()),
    ('scale and encode', ScaleEncode())
])

if __name__ == '__main__':

    current_file = Path(__file__).resolve()
    base_path = current_file.parents[3]  # 0 is file, 1 is parent, ..., 3 = three levels up
    base_path = str(base_path)
    client_data_train = pd.read_csv(base_path + '/data/raw/train/train_client_data_raw.csv')
    price_data_train = pd.read_csv(base_path + '/data/raw/train/train_price_data_raw.csv')
    teste = pipeline.fit_transform([client_data_train, price_data_train])
    
    