from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from prince import MCA
import pandas as pd

class DFMCA(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, prefix='mca_', **kwargs):
        self.columns        = columns
        self.prefix         = prefix
        self.model          = MCA(**kwargs)
        self.transform_cols = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.model.fit(X[self.transform_cols])

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = self.model.transform(X[self.transform_cols])
        new_X.rename(columns=dict(zip(new_X.columns, [f'{self.prefix}{x}' for x in new_X.columns])), inplace=True)

        new_X = pd.concat([X, new_X], axis=1)
        new_X.drop(columns=self.transform_cols, inplace=True)

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)