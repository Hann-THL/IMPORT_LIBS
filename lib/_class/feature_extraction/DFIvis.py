from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from ivis import Ivis
import pandas as pd

# Reference:
# - https://bering-ivis.readthedocs.io/en/latest/supervised.html
# - https://bering-ivis.readthedocs.io/en/latest/hyperparameters.html

# Observations    k         n_epochs_without_progress    model
# --------------------------------------------------------------
# < 1K            10-15     20-30                        maaten
# 1K-10K          10-30     10-20                        maaten
# 10K-50K         15-150    10-20                        maaten
# 50K-100K        15-150    10-15                        maaten
# 100K-500K       15-150    5-10                         maaten
# 500K-1M         15-150    3-5                          szubert
# > 1M            15-150    2-3                          szubert

# model     hidden layer
# -------------------------------
# szubert   128  -> 128  -> 128
# hinton    2000 -> 1000 -> 500
# maaten    500  -> 500  -> 2000

# supervision_metric
# ------------------
# - Classification:
#   - binary_crossentropy, hinge, squared_hinge
# - Classification (Multi-Class):
#   - sparse_categorical_crossentropy, categorical_crossentropy, categorical_hinge
# - Regression:
#   - mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error

# distance
# ---------------------------------
# - euclidean, pn
# - softmax_ratio, softmax_ratio_pn
# - manhattan, manhattan_pn
# - chebyshev, chebyshev_pn
# - cosine, cosine_pn

class DFIvis(BaseEstimator, TransformerMixin):
    # NOTE:
    # - DFIvis(embedding_dims=df.shape[1]) to remain every dimensions
    def __init__(self, columns=None, prefix='ivis_', **kwargs):
        self.columns        = columns
        self.prefix         = prefix
        self.model          = Ivis(**kwargs)
        self.transform_cols = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]
        self.model.fit(X[self.transform_cols].values, y.values if y is not None else y)

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = pd.DataFrame(
            self.model.transform(X[self.transform_cols].values),
            columns=[f'{self.prefix}{x}' for x in range(self.model.embedding_dims)]
        )
        new_X = pd.concat([X.drop(columns=self.transform_cols), new_X], axis=1)

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)