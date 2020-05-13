from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

class DFROCAUCThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, threshold=.5, estimator=RandomForestClassifier()):
        self.columns        = columns
        self.threshold      = threshold
        self.estimator      = estimator
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]

        # Separate dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        # Univariate ROC-AUC
        scores = []
        for column in self.transform_cols:
            self.estimator.fit(X_train[[column]], y_train)
            y_pred = self.estimator.predict(X_test[[column]])
            scores.append(roc_auc_score(y_test, y_pred))

        self.stat_df = pd.DataFrame({
            'feature': X[self.transform_cols].columns,
            'roc_auc': scores,
            'support': np.where(np.array(scores) > self.threshold, True, False)
        })

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        features = self.stat_df[self.stat_df['support']]['feature'].values
        new_X    = X[features].copy()

        return new_X
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)