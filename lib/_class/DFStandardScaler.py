from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd

class DFStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, **kwargs):
        self.columns        = columns
        self.transform_cols = []
        self.model          = StandardScaler(**kwargs)
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.model.fit(X[self.transform_cols])
    
    def transform(self, X):
        new_X = X.copy()
        new_X[self.transform_cols] = self.model.transform(X[self.transform_cols])

        return new_X
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        new_X = X.copy()
        new_X[self.transform_cols] = self.model.inverse_transform(X[self.transform_cols])

        return new_X