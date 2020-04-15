from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from scipy.special import boxcox, inv_boxcox

# Reference: https://www.statisticshowto.com/box-cox-transformation/
class DFBoxCoxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, lmbda=0):
        self.columns        = columns
        self.lmbda          = lmbda
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y=None):
        self.transform_cols = [x for x in X.columns if x in self.columns]

        # Reference: https://help.gooddata.com/doc/en/reporting-and-dashboards/maql-analytical-query-language/maql-expression-reference/aggregation-functions/statistical-functions/predictive-statistical-use-cases/normality-testing-skewness-and-kurtosis
        # Highly skewed:           -1   > Skewness > 1
        # Moderate skewed:         -0.5 < Skewness < -1
        #                           0.5 < Skewness < 1
        # Approximately symmetric: -0.5 < Skewness < 0.5
        skew_df      = X[self.transform_cols].skew().to_frame(name='Skewness')
        # Normal distributed kurtosis: 3
        kurt_df      = X[self.transform_cols].kurt().to_frame(name='Kurtosis')
        self.stat_df = skew_df.merge(kurt_df, left_index=True, right_index=True, how='left')

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        for column in self.transform_cols:
            new_X[column] = boxcox(new_X[column], self.lmbda)

        # Transformed skewness & kurtosis
        skew_df      = new_X[self.transform_cols].skew().to_frame(name='Skewness (Box Cox)')
        kurt_df      = new_X[self.transform_cols].kurt().to_frame(name='Kurtosis (Box Cox)')
        stat_df      = skew_df.merge(kurt_df, left_index=True, right_index=True, how='left')
        self.stat_df = self.stat_df.merge(stat_df, left_index=True, right_index=True, how='left')

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).self.transform(X)
    
    def inverse_transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        for column in self.transform_cols:
            new_X[column] = inv_boxcox(new_X[column], self.lmbda)

        return new_X