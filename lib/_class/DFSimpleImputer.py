from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer

# Reference: https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779
class DFSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, **kwargs):
        self.columns        = columns
        self.model          = SimpleImputer(**kwargs)
        self.transform_cols = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.model.fit(X[self.transform_cols])

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        new_X[self.transform_cols] = self.model.transform(X[self.transform_cols])

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).self.transform(X)