from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from category_encoders import BinaryEncoder
import pandas as pd

class DFBinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, **kwargs):
        self.columns        = columns
        self.model          = BinaryEncoder(**kwargs)
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
        new_X[new_X.columns] = new_X[new_X.columns].astype('int8')

        new_X = pd.concat([X, new_X], axis=1)
        new_X.drop(columns=self.transform_cols, inplace=True)

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        columns = [x for x in X.columns if any([y for y in self.transform_cols if x.startswith(f'{y}_')])]
        new_X   = self.model.inverse_transform(X[columns])

        new_X   = pd.concat([X, new_X], axis=1)
        new_X.drop(columns=columns, inplace=True)

        return new_X