from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
import pandas as pd
import numpy as np
from tqdm import tqdm

class DFMSEThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, k=None, estimator=RandomForestRegressor(), cv=RepeatedKFold()):
        self.columns        = columns
        self.k              = k
        self.estimator      = estimator
        self.cv             = cv
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]

        # Univariate MSE
        cv_scores = []
        for column in self.transform_cols:
            scores = []
            splits = tqdm(self.cv.split(X, y))

            for train_index, test_index in splits:
                X_train = X.loc[train_index][[column]]
                y_train = y.loc[train_index]
                X_test  = X.loc[test_index][[column]]
                y_test  = y.loc[test_index]

                self.estimator.fit(X_train, y_train)
                y_pred = self.estimator.predict(X_test)
                scores.append(round(mean_squared_error(y_test, y_pred), 5))
                splits.set_description(f'Cross-Validation[{column}]')

            cv_scores.append(scores)

        self.stat_df = pd.DataFrame({
            'feature':  self.transform_cols,
            'cv_score': cv_scores
        })
        self.stat_df['average_score'] = self.stat_df['cv_score'].apply(lambda x: np.mean(x))
        self.stat_df['support']       = self.stat_df['average_score'] <= self.stat_df['average_score'].mean()

        # K features with lowest score (MSE)
        if self.k is not None:
            rank_df = self.stat_df.copy()
            rank_df['k_support'] = True
            rank_df.sort_values(by='average_score', inplace=True)
            rank_df = rank_df[['feature', 'k_support']][:self.k]

            self.stat_df = self.stat_df.merge(rank_df, on='feature', how='left')
            self.stat_df['k_support'].fillna(False, inplace=True)

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        support  = 'support' if self.k is None else 'k_support'
        features = self.stat_df[self.stat_df[support]].sort_values(by='average_score')['feature'].values
        new_X    = X[features].copy()

        return new_X
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)