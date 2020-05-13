from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import pandas as pd

class DFLassoSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, k=None, **kwargs):
        self.columns        = columns
        self.k              = k
        self.selector       = SelectFromModel(Lasso(**kwargs))
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.selector.fit(X[self.transform_cols], y)

        # Irrelevant features are with zero coefficient
        self.stat_df = pd.DataFrame({
            'feature':     X[self.transform_cols].columns,
            'coefficient': self.selector.estimator_.coef_,
            'support':     self.selector.get_support()
        })

        # K features with highest score
        if self.k is not None:
            rank_df = self.stat_df.copy()
            rank_df['k_support'] = True
            rank_df.sort_values(by='coefficient', ascending=False, inplace=True)
            rank_df = rank_df[['feature', 'k_support']][:self.k]

            self.stat_df = self.stat_df.merge(rank_df, on='feature', how='left')
            self.stat_df['k_support'].fillna(False, inplace=True)

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        if self.k is None:
            new_X = pd.DataFrame(
                self.selector.transform(X[self.transform_cols]),
                columns=self.stat_df[self.stat_df['support']]['feature'].values)
        else:
            features = self.stat_df[self.stat_df['k_support']].sort_values(by='coefficient', ascending=False)['feature'].values
            new_X    = X[features].copy()

        return new_X
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)