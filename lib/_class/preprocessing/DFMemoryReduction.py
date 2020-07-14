from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from tqdm import tqdm

class DFMemoryReduction(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.difference_df  = None

    def fit(self, X, y=None):
        # Passthrough only
        return self

    def transform(self, X):
        new_X, self.difference_df = self.__reduce_memory_usage(X)

        return new_X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    # Reference:
    # - https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    # - https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    def __reduce_memory_usage(self, df):
        def mb_memory(df):
            return df.memory_usage().sum() / 1024**2
        
        new_df = df.copy()

        for column in tqdm(new_df.select_dtypes(include=['number', 'object']).columns):
            dtype = new_df[column].dtype.name.lower()

            # Object fields
            if dtype == 'object':
                new_df[column] = new_df[column].astype('category')
                continue

            for downcast in ['unsigned', 'signed', 'integer', 'float']:
                series = pd.to_numeric(new_df[column], downcast=downcast)
                if series.dtype.name.lower() != dtype:
                    new_df[column] = series
                    break

        initial_memory  = mb_memory(df)
        optimize_memory = mb_memory(new_df)
        print(f'Initial memory usage:   {initial_memory :.2f} MB')
        print(f'Optimized memory usage: {optimize_memory :.2f} MB')
        print(f'Memory optimized by {(initial_memory - optimize_memory) / initial_memory * 100:.2f} %')

        difference_df = df.select_dtypes('number') - new_df.select_dtypes('number')
        difference_df = pd.concat([difference_df.mean(), difference_df.std()], axis=1)
        difference_df.rename(columns={
            0: 'Mean',
            1: 'Std',
        }, inplace=True)
        
        return new_df, difference_df