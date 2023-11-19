import numpy as np
from scipy.stats import laplace
import math


class ClassificationEstimator(object):
    """ 
    Classification estimator that record the privatised labels. 
    
    Parameters
    ----------
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
        
    epsilon : float
        Privacy budget. 
        
    lamda : float
        
    Attributes
    ----------
    
    U_P: list
    
    V_P: list
    
    U_Q: list
    
    V_Q: list
    

    """
    def __init__(self, 
                 max_depth,
                 ancestor_depth,
                 epsilon,
                 lamda,
                 X_range,
                 lepski_ratio = 1,
                 n_Q = None,
                 ):
        
        self.max_depth = max_depth
        self.ancestor_depth = ancestor_depth
        self.n_Q = n_Q
        self.epsilon = epsilon
        self.noise_level = 4 / epsilon
        self.lamda = lamda
        self.X_range = X_range
        self.if_pruned = 0 
        self.lepski_ratio = lepski_ratio
        

    def fit(self, dt_X, dt_Y):

        self.U_Q = dt_Y.shape[0]
        self.V_Q = dt_Y.sum()
        self.dim = dt_X.shape[1]
        
        return self
        
        
    def get_data(self, Y, in_idx):
        
        noise_U = laplace.rvs(size = in_idx.shape[0], scale = self.noise_level)
        noise_V = laplace.rvs(size = in_idx.shape[0], scale = self.noise_level)

        self.V_P = (Y.ravel() * in_idx.ravel() + noise_V).sum()
        self.U_P = (            in_idx.ravel() + noise_U).sum()
        
        self.n_P = Y.shape[0]
        
        self.true_V_P = (Y.ravel() * in_idx.ravel() ).sum()
        self.true_U_P = (            in_idx.ravel() ).sum()
        
        return self

           
    
    def test_statistic(self):
        

        
            
        k = self.max_depth - self.ancestor_depth
        c2 = np.log(self.n_P + self.n_Q) + k + 1
        c2q = np.log(self.n_Q) + k + 1
        c1 = 2**(self.ancestor_depth + 1) * (2 * np.log(2 * self.n_P) + self.epsilon * 2**( - k / 2) * c2**0.5)**2
        u = c1 / c2 * self.n_P / self.epsilon**2

        
        if self.lamda is not None:
            self.eta = (self.V_P + self.lamda * self.V_Q) / (self.U_P + self.lamda * self.U_Q)
            r = c2**0.5 * (u + self.lamda**2 * self.U_Q)**0.5   / (self.U_P + self.lamda * self.U_Q)
        else:
            if self.U_Q == 0:
                self.eta = 1 / 2
                self.y_hat = 1
                return 0
            signal_P = self.V_P / self.U_P - 1 / 2
            signal_Q = self.V_Q / self.U_Q - 1 / 2
            
            if self.U_P <= 0 or self.V_P <= 0:
                self.eta = self.V_Q / self.U_Q
                r =  c2q**0.5 / self.U_Q**0.5  
            else:
                truncated_V_P = self.V_P
                if self.V_P > self.U_P:
                    signal_P = 1 / 2
                    truncated_V_P = self.U_P
                elif 0 <= self.V_P / self.U_P <= 1:
                    if u > self.U_P:
                        self.eta = self.V_Q / self.U_Q
                        r =  c2q**0.5 / self.U_Q**0.5  
                        self.y_hat = (self.eta > 1 / 2).astype(int)
                        return np.abs(self.eta - 0.5) / r
                    else:
                        u = self.U_P
                else:
                    raise ValueError
                
                if signal_P * signal_Q > 0:
                    self.lamda = u  / self.U_P * signal_Q / signal_P
                    self.eta = (truncated_V_P + self.lamda * self.V_Q) / (self.U_P + self.lamda * self.U_Q)
                    r = c2**0.5 * (u + self.lamda**2 * self.U_Q)**0.5   / (self.U_P + self.lamda * self.U_Q) 
                    if np.abs(self.eta - 0.5) / r < np.abs(self.V_Q / self.U_Q - 1/2) /  c2q**0.5 * self.U_Q**0.5:
                        self.eta = self.V_Q / self.U_Q
                        r =  c2q**0.5 / self.U_Q**0.5  
                else:
                    test_statistic_P = np.abs(truncated_V_P / self.U_P - 1/2) / c2**0.5 / u**0.5 * self.U_P
                    test_statistic_Q = np.abs(self.V_Q / self.U_Q - 1/2) /  c2q**0.5 * self.U_Q**0.5
                    if test_statistic_P < test_statistic_Q:
                        self.eta = self.V_Q /  self.U_Q
                        r =  c2q**0.5 / self.U_Q**0.5 
                    else:
                        self.eta = truncated_V_P / self.U_P
                        r = c2**0.5 * u**0.5 / self.U_P
        
        self.y_hat = (self.eta > 1 / 2).astype(int)
        return np.abs(self.eta - 0.5) / r
    
    
    def predict(self, test_X):
  
        y_predict = np.full(test_X.shape[0], self.y_hat)
        return y_predict
    
    def separate_predict(self, test_X):
        
        y_predict_P = np.full(test_X.shape[0], int(self.V_P / self.U_P > 1/2))
        y_predict_Q = np.full(test_X.shape[0], int(self.V_Q / self.U_Q > 1/2))
        return y_predict_P, y_predict_Q
    
    def predict_proba(self, test_X):
  
        y_predict = np.full(test_X.shape[0], self.eta)
        return y_predict
    
    def separate_predict_proba(self, test_X):
        
        y_predict_proba_P = np.full(test_X.shape[0], self.V_P / self.U_P )
        y_predict_proba_Q = np.full(test_X.shape[0], self.V_Q / self.U_Q )
        return y_predict_proba_P, y_predict_proba_Q
    
    
    
    

class AncestorNodePruningEstimator(ClassificationEstimator):
    """ 
    Estimator at ancestor node for pruning the tree structure
    
    Parameters
    ----------
    
    depth : int
    
    epsilon : float
    
    lamda : float
     
    Attributes
    ----------
    U_P: list
    
    V_P: list
    
    U_Q: list
    
    V_Q: list

    """
    def __init__(self,
                 max_depth,
                 ancestor_depth,
                 epsilon,
                 lamda,
                 X_range = None,
                 lepski_ratio = 1,
                 ):
        super(AncestorNodePruningEstimator, self).__init__(
                 max_depth = max_depth,
                 ancestor_depth = ancestor_depth,
                 epsilon = epsilon,
                 lamda   = lamda,
                 X_range = X_range,
                 lepski_ratio = lepski_ratio,
        )
        

        self.if_pruned = 1
        
    def get_data(self, node_list):
        self.node_list = node_list
        
        self.n_P = node_list[0].n_P
        self.dim = node_list[0].dim
        self.n_Q = node_list[0].n_Q
        self.V_P = np.sum([node.V_P for node in node_list])
        self.V_Q = np.sum([node.V_Q for node in node_list])
        self.U_P = np.sum([node.U_P for node in node_list])
        self.U_Q = np.sum([node.U_Q for node in node_list])
        self.true_U_P = np.sum([node.true_U_P for node in node_list])
        self.true_V_P = np.sum([node.true_V_P for node in node_list])
        
        
        
        merged_X_range = np.concatenate([node.X_range for node in node_list], axis = 0)
        self.X_range = np.array([np.min(merged_X_range, axis = 0), np.max(merged_X_range, axis = 0)])
        
   
        
    