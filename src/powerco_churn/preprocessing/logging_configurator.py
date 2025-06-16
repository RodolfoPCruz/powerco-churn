class LoggingConfigurator(BaseEstimator, TransformerMixin):
    def __init__(self, log_file_name = 'pipeline.log', level = logging.INFO):
        self.log_file_name = log_file_name
        self.level = level
        
    def fit(self, X, y=None):
        configure_logging(log_file_name=self.log_file_name, level=self.level)
        return self

    def transform(self, X):
        return X  # Do nothing, just pass the data through