from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np
from scipy.stats import linregress

class DFRegressionThreshold(BaseEstimator, TransformerMixin):
    # Reference: https://www.dummies.com/education/math/statistics/what-a-p-value-tells-you-about-statistical-data/
    # p-value lesser than 0.05 indicates strong evidence against null hypothesis
    def __init__(self, target, r2_threshold=.2, p_threshold=.05):
        self.target       = target
        self.r2_threshold = r2_threshold
        self.p_threshold  = p_threshold
        self.stat_df      = None
        
    def fit(self, X, y=None):
        self.stat_df = pd.DataFrame({
            'feature': [x for x in X.select_dtypes(include='number') if x != self.target]
        })

        for row in self.stat_df.itertuples():
            slope, intercept, r_value, p_value, std_err = linregress(X[row.feature], X[self.target])

            self.stat_df.at[row.Index, 'r_value'] = r_value
            self.stat_df.at[row.Index, 'p_value'] = p_value
        self.stat_df['r2'] = np.square(self.stat_df['r_value'])
    
    def transform(self, X):
        if self.stat_df is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X     = X.copy()
        remain_df = self.stat_df[(np.round(self.stat_df['r2'], 2) >= self.r2_threshold) & (np.round(self.stat_df['p_value'], 2) <= self.p_threshold)].copy()
        remain_df.sort_values(by='r2', ascending=False, inplace=True)

        return new_X[list(remain_df['feature']) + [self.target]]
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)