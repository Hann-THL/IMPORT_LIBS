from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np

# Reference: https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
class DFMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, weight=0, columns=None, concat_symbol='|', decimal=5):
        self.weight                = weight
        self.columns               = columns
        self.concat_symbol         = concat_symbol
        self.decimal               = decimal
        self.transform_mapper_dict = None
        
    def fit(self, X, y):
        self.columns               = X.columns if self.columns is None else self.columns
        self.transform_mapper_dict = {}

        df     = pd.concat([X, y], axis=1)
        target = y.name

        for column in self.columns:
            agg_df        = df.groupby(column)[target].agg(['count', 'mean'])
            count         = agg_df['count']
            estimate_mean = agg_df['mean']
            overall_mean  = df[target].mean()
            smooth_mean   = (count * estimate_mean + self.weight * overall_mean) / (count + self.weight)

            mapper      = np.round(smooth_mean, self.decimal)
            mapper.name = 'encode'

            self.transform_mapper_dict[column] = mapper.sort_values(ascending=False)

        return self
    
    def transform(self, X):
        if self.transform_mapper_dict is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        for k,v in self.transform_mapper_dict.items():
            new_X[k] = new_X[k].map(v)

        return new_X
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
    
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