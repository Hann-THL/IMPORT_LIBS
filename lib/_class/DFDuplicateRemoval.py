from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

class DFDuplicateRemoval(BaseEstimator, TransformerMixin):
    def __init__(self, keep='first', target=None, subset=None):
        keeps = ['first', 'last', 'mean']
        assert keep in keeps, f'keep not in valid list: {keeps}'

        if keep == 'mean':
            assert target is not None, 'target is not provided.'

        self.keep         = keep
        self.target       = target
        self.subset       = subset
        self.duplicate_df = None
        
    def fit(self, X, y=None):
        self.subset = X.columns if self.subset is None else self.subset
        self.subset = [x for x in self.subset if x != self.target]

        self.duplicate_df = X[X.duplicated(subset=[x for x in X.columns if x in self.subset], keep=False)].copy()
        self.duplicate_df = self.duplicate_df.sort_values(by=list(self.duplicate_df.columns))

        return self
    
    def transform(self, X):
        if self.duplicate_df is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        if self.keep == 'mean':
            new_X = new_X.groupby(self.subset).agg(
                DUPLICATE_REMOVAL_mean=(self.target, 'mean')
            ).reset_index()
            new_X.rename(columns={'DUPLICATE_REMOVAL_mean': self.target}, inplace=True)

        else:
            new_X = new_X.drop_duplicates(subset=self.subset, keep=self.keep).reset_index(drop=True)

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).self.transform(X)