from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
import numpy as np
from tqdm import tqdm

class DFROCAUCThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, threshold=.5, estimator=RandomForestClassifier(), cv=RepeatedStratifiedKFold(), multi_class='raise'):
        self.columns        = columns
        self.threshold      = threshold
        self.estimator      = estimator
        self.cv             = cv
        self.multi_class    = multi_class
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]

        # Univariate ROC-AUC
        cv_scores = []
        for column in self.transform_cols:
            scores = []
            splits = tqdm(self.cv.split(X, y))

            for train_index, test_index in splits:
                X_train = X.loc[train_index][[column]]
                y_train = y.loc[train_index]
                X_test  = X.loc[test_index][[column]]
                y_test  = y.loc[test_index]

                if self.multi_class in ['ovo', 'ovr']:
                    y_train = pd.get_dummies(y_train)
                    y_test  = pd.get_dummies(y_test)

                self.estimator.fit(X_train, y_train)
                y_pred = self.estimator.predict(X_test)
                scores.append(round(roc_auc_score(y_test, y_pred, multi_class=self.multi_class), 5))
                splits.set_description(f'Cross-Validation[{column}]')

            cv_scores.append(scores)

        self.stat_df = pd.DataFrame({
            'feature':  self.transform_cols,
            'cv_score': cv_scores
        })
        self.stat_df['average_score'] = self.stat_df['cv_score'].apply(lambda x: np.mean(x))
        self.stat_df['support']       = np.where(np.array(self.stat_df['average_score']) > self.threshold, True, False)

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        features = self.stat_df[self.stat_df['support']].sort_values(by='average_score', ascending=False)['feature'].values
        new_X    = X[features].copy()

        return new_X
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)