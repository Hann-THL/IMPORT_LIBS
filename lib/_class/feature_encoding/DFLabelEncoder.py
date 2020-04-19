from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class DFLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns        = columns
        self.model_dict     = None
        
    def fit(self, X, y=None):
        self.columns    = X.columns if self.columns is None else self.columns
        self.model_dict = {}

        for column in self.columns:
            model = LabelEncoder()
            model.fit(X[column])

            self.model_dict[column] = model

        return self
    
    def transform(self, X):
        if self.model_dict is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        for k,v in self.model_dict.items():
            new_X[k] = v.transform(new_X[k])

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        if self.model_dict is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        for k,v in self.model_dict.items():
            new_X[k] = v.inverse_transform(new_X[k])

        return new_X