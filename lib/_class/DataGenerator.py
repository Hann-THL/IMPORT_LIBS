import numpy as np

from tensorflow.keras.utils import Sequence

# Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        batch_X = self.X[index * self.batch_size : (index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size : (index + 1) * self.batch_size]
        
        return batch_X, batch_y