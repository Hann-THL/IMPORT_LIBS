from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from category_encoders import BackwardDifferenceEncoder
import pandas as pd

class DFBackwardDifferenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, **kwargs):
        self.columns        = columns
        self.model          = BackwardDifferenceEncoder(**kwargs)
        self.transform_cols = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.model.fit(X[self.transform_cols])

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.drop(columns=self.transform_cols)
        new_X = pd.concat([
            new_X,
            self.model.transform(X[self.transform_cols])
        ], axis=1)

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)