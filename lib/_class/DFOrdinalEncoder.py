from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import pandas as pd

class DFOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapper_dict, concat_symbol='|'):
        self.mapper_dict           = mapper_dict
        self.concat_symbol         = concat_symbol
        self.transform_mapper_dict = None
        
    def fit(self, X, y=None):
        self.transform_mapper_dict = {k: v for k,v in self.mapper_dict.items() if k in X.columns}
        return self
    
    def transform(self, X):
        if self.transform_mapper_dict is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X = X.copy()
        for k,v in self.transform_mapper_dict.items():
            new_X[k] = new_X[k].map(v)

        return new_X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).self.transform(X)
    
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