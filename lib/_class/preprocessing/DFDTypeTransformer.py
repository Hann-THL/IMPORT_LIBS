from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import pandas as pd

class DFDTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dtype_dict):
        self.dtype_dict           = dtype_dict
        self.transform_dtype_dict = None
        
    def fit(self, X, y=None):
        self.transform_dtype_dict = {k: [x for x in v if x in X.columns] for k,v in self.dtype_dict.items()}

        return self
    
    def transform(self, X):
        if self.transform_dtype_dict is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        for k,v in self.transform_dtype_dict.items():
            new_X[v] = new_X[v].astype(k)

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)