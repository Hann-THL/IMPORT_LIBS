from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

class DFVarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, **kwargs):
        self.columns        = columns
        self.model          = VarianceThreshold(**kwargs)
        self.transform_cols = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.model.fit(X[self.transform_cols])

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = pd.DataFrame(
            self.model.transform(X[self.transform_cols]),
            columns=np.array(self.transform_cols)[self.model.get_support()]
        )
        new_X = pd.concat([X.drop(columns=self.transform_cols), new_X], axis=1)

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)