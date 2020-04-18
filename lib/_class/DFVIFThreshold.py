from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Reference: https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
class DFVIFThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=5, columns=None, exclude_const=False, show_progress=False):
        self.threshold      = threshold
        self.columns        = columns
        self.exclude_const  = exclude_const
        self.show_progress  = show_progress
        self.transform_cols = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        # Reference: https://www.statisticshowto.com/variance-inflation-factor/
        tmp_X = X.copy()
        while 1:
            vif_df  = self.calc_vif(tmp_X)
            index   = vif_df[~vif_df.index.isin(['const'])]['VIF'].idxmax()
            max_vif = vif_df.at[index, 'VIF']

            if max_vif <= self.threshold:
                break
            tmp_X.drop(columns=[index], inplace=True)

            if self.show_progress:
                print(f'Removed: {index} ({max_vif :.5f})')

        new_X = X[[x for x in tmp_X.columns if x in X.columns]].copy()
        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def calc_vif(self, X):
        tmp_X  = X[[x for x in X.columns if x in self.transform_cols]].copy()
        tmp_X  = tmp_X if self.exclude_const else add_constant(tmp_X)
        vif_df = pd.DataFrame(
            [variance_inflation_factor(tmp_X.values, i) for i in range(tmp_X.shape[1])],
            index=tmp_X.columns,
            columns=['VIF'])

        return vif_df