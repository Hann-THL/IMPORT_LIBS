from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

class DFMSEThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, k=None, estimator=RandomForestRegressor()):
        self.columns        = columns
        self.k              = k
        self.estimator      = estimator
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]

        # Separate dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Univariate MSE
        scores = []
        for column in self.transform_cols:
            self.estimator.fit(X_train[[column]], y_train)
            y_pred = self.estimator.predict(X_test[[column]])
            scores.append(mean_squared_error(y_test, y_pred))

        self.stat_df = pd.DataFrame({
            'feature': X[self.transform_cols].columns,
            'mse':     scores
        })

        # K features with highest score
        if self.k is None:
            self.stat_df['k_support'] = True
        else:
            rank_df = self.stat_df.copy()
            rank_df['k_support'] = True
            rank_df.sort_values(by='mse', inplace=True)
            rank_df = rank_df[['feature', 'k_support']][:self.k]

            self.stat_df = self.stat_df.merge(rank_df, on='feature', how='left')
            self.stat_df['k_support'].fillna(False, inplace=True)

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        features = self.stat_df[self.stat_df['k_support']]['feature'].values
        new_X    = X[features].copy()

        return new_X
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)