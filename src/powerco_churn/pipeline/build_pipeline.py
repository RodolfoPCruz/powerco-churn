

from powerco_churn.preprocessing.date_processing import DateParserTransformer
from powerco_churn.preprocessing.date_processing import CreatingDateFeatures
from powerco_churn.preprocessing.dropping_missing import DropMissing
from powerco_churn.preprocessing.temporal_series_processing import TransformPricesTemporalSeries




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



if __name__ == '__main__':
    client_data_train_raw = pd.read_csv('../data/raw/train/train_client_data_raw.csv')
    price_data_train_raw = pd.read_csv('../data/raw/train/train_price_data_raw.csv')
