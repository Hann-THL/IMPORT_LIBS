from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from mlxtend.feature_selection import SequentialFeatureSelector
import pandas as pd

class DFSequentialFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, **kwargs):
        self.columns        = columns
        self.selector       = SequentialFeatureSelector(**kwargs)
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.selector.fit(X[self.transform_cols], y)

        self.stat_df = pd.DataFrame.from_dict(self.selector.get_metric_dict()).T
        self.stat_df.at[self.stat_df['avg_score'].astype(float).idxmax(), 'support'] = True
        self.stat_df['support'].fillna(False, inplace=True)

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        features = list(self.stat_df[self.stat_df['support']]['feature_names'].values[0])
        new_X    = X[features].copy()

        return new_X
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)