from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import RepeatedKFold
import pandas as pd
import numpy as np
from tqdm import tqdm

class DFANOVARegressorThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, k=None, cv=RepeatedKFold()):
        self.columns        = columns
        self.k              = k
        self.cv             = cv
        self.selector       = SelectKBest(f_regression, k='all')
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]

        cv_scores   = []
        cv_pvalues  = []
        cv_supports = []
        splits      = tqdm(self.cv.split(X, y))

        for train_index, test_index in splits:
            X_sample = X.loc[np.append(train_index, test_index)][self.transform_cols]
            y_sample = y.loc[np.append(train_index, test_index)]
            
            self.selector.fit(X_sample, y_sample)
            cv_scores.append(self.selector.scores_)
            cv_pvalues.append(self.selector.pvalues_)
            cv_supports.append(self.selector.scores_ > self.selector.scores_.mean())
            splits.set_description('Cross-Validation')

        cv_scores    = np.array(cv_scores).T
        cv_pvalues   = np.array(cv_pvalues).T
        cv_supports  = np.array(cv_supports).T
        self.stat_df = pd.DataFrame({
            'feature': self.transform_cols
        })
        self.stat_df['cv_score']       = self.stat_df.index.to_series().apply(lambda x: list(np.round(cv_scores[x], 5)))
        self.stat_df['cv_pvalue']      = self.stat_df.index.to_series().apply(lambda x: list(np.round(cv_pvalues[x], 5)))
        self.stat_df['cv_support']     = self.stat_df.index.to_series().apply(lambda x: list(cv_supports[x]))
        self.stat_df['average_score']  = self.stat_df['cv_score'].apply(lambda x: np.mean(x))
        self.stat_df['average_pvalue'] = self.stat_df['cv_pvalue'].apply(lambda x: np.mean(x))
        self.stat_df['support']        = self.stat_df['cv_support'].apply(lambda x: np.where(np.sum(np.where(x, 1, 0)) / len(x) > .5, True, False))
        self.stat_df['support']        = np.where(self.stat_df['average_pvalue'] > .05, False, self.stat_df['support'])

        # K features with highest score
        if self.k is not None:
            rank_df = self.stat_df.copy()
            rank_df['k_support'] = True
            rank_df.sort_values(by='average_score', ascending=False, inplace=True)
            rank_df = rank_df[['feature', 'k_support']][:self.k]

            self.stat_df = self.stat_df.merge(rank_df, on='feature', how='left')
            self.stat_df['k_support'].fillna(False, inplace=True)

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        support  = 'support' if self.k is None else 'k_support'
        features = self.stat_df[self.stat_df[support]].sort_values(by='average_score', ascending=False)['feature'].values
        new_X    = X[features].copy()

        return new_X
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)