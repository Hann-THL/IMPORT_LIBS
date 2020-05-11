from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

class DFDuplicateThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns        = columns
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]

        self.stat_df = X.T.duplicated().to_frame('duplicate').reset_index()
        self.stat_df.rename(columns={'index': 'feature'}, inplace=True)

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        new_X.drop(columns=self.stat_df[self.stat_df['duplicate']]['feature'], inplace=True)

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)