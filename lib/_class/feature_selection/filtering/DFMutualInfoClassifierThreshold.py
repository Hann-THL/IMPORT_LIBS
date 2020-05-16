from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
import numpy as np
from tqdm import tqdm

class DFMutualInfoClassifierThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, k='all', cv=RepeatedStratifiedKFold()):
        self.columns        = columns
        self.selector       = SelectKBest(mutual_info_classif, k=k)
        self.cv             = cv
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]

        cv_scores   = []
        cv_supports = []
        splits      = tqdm(self.cv.split(X, y))

        for train_index, test_index in splits:
            X_sample = X.loc[np.append(train_index, test_index)][self.transform_cols]
            y_sample = y.loc[np.append(train_index, test_index)]
            
            self.selector.fit(X_sample, y_sample)
            cv_scores.append(self.selector.scores_)
            cv_supports.append(self.selector.get_support())
            splits.set_description('Cross-Validation')

        cv_scores    = np.array(cv_scores).T
        cv_supports  = np.array(cv_supports).T
        self.stat_df = pd.DataFrame({
            'feature': self.transform_cols
        })
        self.stat_df['cv_score']      = self.stat_df.index.to_series().apply(lambda x: list(np.round(cv_scores[x], 5)))
        self.stat_df['cv_support']    = self.stat_df.index.to_series().apply(lambda x: list(cv_supports[x]))
        self.stat_df['average_score'] = self.stat_df['cv_score'].apply(lambda x: np.mean(x))
        self.stat_df['support']       = self.stat_df['cv_support'].apply(lambda x: np.where(np.sum(np.where(x, 1, 0)) / len(x) > .5, True, False))

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        features = self.stat_df[self.stat_df['support']].sort_values(by='average_score', ascending=False)['feature'].values
        new_X    = X[features].copy()

        return new_X
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)