from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from ivis import Ivis
import pandas as pd

# Reference: https://bering-ivis.readthedocs.io/en/latest/supervised.html
class DFIvis(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, prefix='ivis_', **kwargs):
        self.columns        = columns
        self.prefix         = prefix
        self.model          = Ivis(**kwargs)
        self.transform_cols = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.model.fit(X[self.transform_cols].values, y.values if y is not None else y)

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = pd.DataFrame(
            self.model.transform(X[self.transform_cols].values),
            columns=[f'{self.prefix}{x}' for x in range(self.model.embedding_dims)]
        )

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)