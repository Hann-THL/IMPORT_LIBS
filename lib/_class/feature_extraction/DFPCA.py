from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from prince import PCA
import pandas as pd
import numpy as np

class DFPCA(BaseEstimator, TransformerMixin):
    # NOTE:
    # - DFPCA(n_components=df.shape[1]) to remain every dimensions
    # - DFPCA(rescale_with_mean=False, rescale_with_std=False) to avoid using built-in StandardScaler()
    def __init__(self, columns=None, prefix='pca_', **kwargs):
        self.columns        = columns
        self.prefix         = prefix
        self.model          = PCA(**kwargs)
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.model.fit(X[self.transform_cols])

        # Reference: Reference: https://www.appliedaicourse.com/lecture/11/applied-machine-learning-online-course/2896/pca-for-dimensionality-reduction-not-visualization/0/free-videos
        self.stat_df = pd.DataFrame({
            'dimension': [x+1 for x in range(len(self.model.eigenvalues_))],
            'eigenvalues': self.model.eigenvalues_,
            'explained_inertia': self.model.explained_inertia_,
            'cumsum_explained_inertia': np.cumsum(self.model.explained_inertia_)
        })

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = self.model.transform(X[self.transform_cols])
        new_X.rename(columns=dict(zip(new_X.columns, [f'{self.prefix}{x}' for x in new_X.columns])), inplace=True)
        new_X = pd.concat([X.drop(columns=self.transform_cols), new_X], axis=1)

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)