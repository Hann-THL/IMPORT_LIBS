from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

class DFMultiLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.model          = MultiLabelBinarizer(**kwargs)
        self.transform_cols = None
        
    def fit(self, y):
        self.transform_cols = [x for x in y.columns]
        self.model.fit(y[self.transform_cols].values)

        return self
    
    def transform(self, y):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_y = pd.DataFrame(
            self.model.transform(y[self.transform_cols].values),
            columns=[f'MLB_{x}' for x in self.model.classes_]
        )

        return new_y
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        if self.transform_cols is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        new_X   = pd.DataFrame(self.model.inverse_transform(y.values), columns=self.transform_cols)

        return new_X