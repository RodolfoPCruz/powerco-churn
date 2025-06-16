class scale_encode(BaseEstimator, TransformerMixin):
    
    def __init__(self):

        self.scaler = StandardScaler()
        self.onehotencoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')

    def fit(self, X, y = None):

        self.numerical_features_ = [feature for feature in X.columns if pd.api.types.is_numeric_dtype(X[feature])
                                   and X[feature].nunique() > 2] 
        self.categorical_features_ = [feature for feature in X.columns if  pd.api.types.is_object_dtype(X[feature]) 
                                                             or isinstance(X[feature], pd.CategoricalDtype)] 
        self.scaler.fit(X[self.numerical_features_])
        self.onehotencoder.fit(X[self.categorical_features_])
        
        return self

    def transform(self, X):

        check_is_fitted(self, ['numerical_features_', 'categorical_features_'])
        X_copy = X.copy()
        X_copy[self.numerical_features_] = self.scaler.transform(X_copy[self.numerical_features_])
        X_copy_encoded = self.onehotencoder.transform(X_copy[self.categorical_features_])
        X_copy = X_copy.drop(columns = self.categorical_features_)
        X_copy = pd.concat([X_copy, X_copy_encoded], axis = 1)

        return X_copy