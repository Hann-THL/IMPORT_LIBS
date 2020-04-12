from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np

# Reference: https://blogs.sas.com/content/iml/2011/04/27/log-transformations-how-to-handle-negative-data-values.html
class DFPositiveTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, min_value=1):
        assert min_value >= 0, 'min_value should be positive.'

        self.columns               = columns
        self.min_value             = min_value
        self.transform_column_dict = None
        
    def fit(self, X, y=None):
        self.transform_column_dict = {}
        for column in [x for x in X.columns if x in self.columns]:
            self.transform_column_dict[column] = np.min(X[column])

        return self
    
    def transform(self, X):
        if self.transform_column_dict is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        for k,v in self.transform_column_dict.items():
            new_X[k] = new_X[k] + self.min_value - v

        return new_X
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        if self.transform_column_dict is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        for k,v in self.transform_column_dict.items():
            new_X[k] = new_X[k] - self.min_value + v

        return new_X