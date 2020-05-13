from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import RFECV
import pandas as pd

class DFRFECV(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, **kwargs):
        self.columns        = columns
        self.selector       = RFECV(**kwargs)
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.selector.fit(X[self.transform_cols], y)

        self.stat_df = pd.DataFrame({
            'feature':    X[self.transform_cols].columns,
            'ranking':    self.selector.ranking_,
            'grid_score': self.selector.grid_scores_,
            'support':    self.selector.get_support()
        })

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        features = self.stat_df[self.stat_df['support']]['feature'].values
        new_X    = X[features].copy()

        return new_X
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)