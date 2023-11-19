import numpy as np
from numba import njit

@njit
def entropy(y):
    """
    Sample entropy.
    """
    if len(y)>0:
        p = np.mean(y)
        if p == 1:
            return 0
        elif p == 0:
            return 0
        else:
            return - (p * np.log2(p) + (1 - p) * np.log2(1 - p) )
    else:
        return 0


@njit
def infogain(X, dt_Y, d, split):
    """Compute gini decrease for one pair of dimension and split point.

    Parameters
    ----------
    
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
 
    d : int in 0, ..., dim - 1
        The splitting dimension.
    
    split : float
        The splitting point.
    


    """
    
    before_ent = entropy(dt_Y)

    left_idx = X[:,d] < split
    right_idx = X[:,d] >= split
    after_ent = (entropy(dt_Y[left_idx]) * left_idx.sum() + entropy(dt_Y[right_idx]) * right_idx.sum()) / len(dt_Y)

   
    
    
    return before_ent - after_ent

