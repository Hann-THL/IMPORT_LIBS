from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np

class DFCorrelationThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, threshold=.9):
        self.columns        = columns
        self.threshold      = threshold
        self.transform_cols = None
        self.stat_df        = None
        
    def fit(self, X, y=None):
        self.columns        = X.columns if self.columns is None else self.columns
        self.transform_cols = [x for x in X.columns if x in self.columns]

        corrmat_df = pd.concat([X[self.transform_cols], y], axis=1).corr().abs()
        corrmat_df = corrmat_df.round(6)

        # Disable correlation of same feature
        for x in corrmat_df.columns:
            corrmat_df.loc[x, x] = np.nan

        # Generate feature stats
        target = None if y is None else y.name
        self.stat_df = pd.DataFrame(index=[x for x in corrmat_df.index if x != target])
        self.stat_df.index.name = 'feature'

        # Identify correlation with target
        if target is None:
            self.stat_df['target_correlation'] = np.nan
        else:
            self.stat_df = self.stat_df.merge(corrmat_df[target].to_frame(), left_index=True, right_index=True, how='left')
            self.stat_df.rename(columns={target: 'target_correlation'}, inplace=True)

        # Identify correlated features
        self.stat_df['feature_correlated']         = self.stat_df.index.to_series().apply(lambda x: [])
        self.stat_df['feature_correlation']        = self.stat_df.index.to_series().apply(lambda x: [])
        self.stat_df['feature_target_correlation'] = self.stat_df.index.to_series().apply(lambda x: [])
        without_target_corrmat_df                  = corrmat_df.copy() if target is None else corrmat_df.drop(columns=[target], index=[target])

        for feature in self.stat_df.index:
            self.stat_df.at[feature, 'feature_correlated'] = list(
                without_target_corrmat_df.index[without_target_corrmat_df.unstack().loc[feature] >= self.threshold]
            )
            self.stat_df.at[feature, 'feature_correlation'] = list(
                without_target_corrmat_df.loc[self.stat_df.at[feature, 'feature_correlated']][feature]
            )
            if target is not None:
                self.stat_df.at[feature, 'feature_target_correlation'] = list(
                    corrmat_df.loc[self.stat_df.at[feature, 'feature_correlated']][target]
                )

        # Threshold 1:
        # - Remain correlated feature which is having highest correlation with target
        self.stat_df['support'] = self.stat_df['feature_target_correlation'].apply(
            lambda x: 0 if len(x) <= 0 else np.max(x)
        ) <= np.where(self.stat_df['target_correlation'].isna(), 1, self.stat_df['target_correlation'])

        # Threshold 2:
        # - Identify correlation matrix for supported features
        # - Drop either one of the correlated features with same correlation with target
        support_corrmat_df = corrmat_df.loc[list(self.stat_df['support']) + ([] if target is None else [False])]
        support_corrmat_df = support_corrmat_df[support_corrmat_df.index]

        # Reference: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
        # Select upper triangle of correlation matrix
        upper_df = support_corrmat_df.where(np.triu(np.ones(support_corrmat_df.shape), k=1).astype(np.bool))

        # Find index of feature columns exceed correlation threshold
        columns = [x for x in upper_df.columns if any(upper_df[x] >= self.threshold)]
        if len(columns) > 0:
            self.stat_df.at[columns, 'support'] = False

        self.stat_df.reset_index(inplace=True)

        return self
    
    def transform(self, X):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        features = self.stat_df[self.stat_df['support']]['feature'].values
        new_X    = X[features].copy()

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)