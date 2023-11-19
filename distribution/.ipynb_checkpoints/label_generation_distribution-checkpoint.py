import numpy as np


class LabelGeneration(object): 
    def __init__(self):
        pass
        
    def apply(self, y_true):
        y = np.random.rand(*y_true.shape) < y_true
        return y.astype(int)
        

