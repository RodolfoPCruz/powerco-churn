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