from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from category_encoders import CatBoostEncoder
import pandas as pd

class DFCatBoostEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, **kwargs):
        self.columns        = columns
        self.model          = CatBoostEncoder(**kwargs)
        self.transform_cols = None
        
    def fit(self, X, y):
        return self.__fit(X, y)
    
    def __fit(self, X, y, fit=True):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]

        if fit:
            self.model.fit(X[self.transform_cols], y)

        return self

    def transform(self, X):
        return self.__transform(X)
    
    def __transform(self, X, y=None):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.drop(columns=self.transform_cols)
        new_X = pd.concat([
            new_X,
            self.model.transform(X[self.transform_cols]) if y is None else self.model.fit_transform(X[self.transform_cols], y)
        ], axis=1)

        return new_X
    
    def fit_transform(self, X, y):
        # NOTE: Result of fit_transform() is different from fit() + transform()
        return self.__fit(X, y, fit=False).__transform(X, y)