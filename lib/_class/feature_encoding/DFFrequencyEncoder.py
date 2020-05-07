from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np

# Reference: https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
class DFFrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, normalize=True, concat_symbol='|', decimal=5):
        self.columns               = columns
        self.normalize             = normalize
        self.concat_symbol         = concat_symbol
        self.decimal               = decimal
        self.transform_mapper_dict = None
        
    def fit(self, X, y=None):
        self.columns               = X.columns if self.columns is None else self.columns
        self.transform_mapper_dict = {}

        for column in self.columns:
            mapper            = np.round(X[column].value_counts(normalize=self.normalize), self.decimal)
            mapper.name       = 'encode'
            mapper.index.name = column

            self.transform_mapper_dict[column] = mapper

        return self
    
    def transform(self, X):
        if self.transform_mapper_dict is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        for k,v in self.transform_mapper_dict.items():
            new_X[k] = new_X[k].map(v).fillna(v.min())

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        if self.transform_mapper_dict is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        for k,v in self.transform_mapper_dict.items():
            mapper_df  = pd.DataFrame(v.items(), columns=['data', 'encode'])
            inverse_df = mapper_df.groupby('encode').agg(
                data=('data', lambda x: self.concat_symbol.join(x))
            ).reset_index()

            inverse_dict = dict(zip(inverse_df['encode'].values, inverse_df['data'].values))
            new_X[k] = new_X[k].map(inverse_dict)

        return new_X