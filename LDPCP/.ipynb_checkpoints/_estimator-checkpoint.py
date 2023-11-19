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
        
        
    Attributes
    ----------
    
    U_P: list
    
    V_P: list
    
    U_Q: list
    
    V_Q: list
    
    weight: float

    """
    def __init__(self, 
                 X_range,
                 epsilon,
                 lamda,
                 ):
        
        
        self.X_range = X_range
        self.noise_level = 4 / epsilon
        self.lamda = lamda
        self.if_pruned = 0 
        
        
        
    def fit(self, dt_X, dt_Y):

        self.U_Q = dt_Y.shape[0]
        self.V_Q = dt_Y.sum()
        
        return self
        
        
    def get_data(self, Y, in_idx):
        
        noise_U = laplace.rvs(size = in_idx.shape[0], scale = self.noise_level)
        noise_V = laplace.rvs(size = in_idx.shape[0], scale = self.noise_level)

        self.V_P = (Y.ravel() * in_idx.ravel() + noise_V).sum()
        self.U_P = (            in_idx.ravel() + noise_U).sum()
        
        # print(self.X_range)
        # print("Y:", (Y.ravel() * in_idx.ravel()).sum(), self.V_P)
        # print("X:",  in_idx.ravel().sum(), self.U_P)
        
        return self

           
    
    def test_statistic(self):
        
        signal_P = self.V_P / self.U_P - 1/2
        signal_Q = self.V_Q / self.U_Q - 1/2
        
        # assign estimation value
        
        if self.lamda is not None:
            self.y_hat = ((self.V_P + self.lamda * self.V_Q) / (self.U_P + self.lamda * self.U_Q) > 1 / 2).astype(int)
            self.eta = (self.V_P + self.lamda * self.V_Q) / (self.U_P + self.lamda * self.U_Q)
        else:
            if self.U_P < 0:
                self.y_hat = (  self.U_Q**0.5  * signal_Q  > 0 ).astype(int)
                self.eta = self.V_Q / self.U_Q
            else:
                self.y_hat = ( self.U_P**0.5 * signal_P + self.U_Q**0.5  * signal_Q  > 0 ).astype(int)
                self.eta = ( self.V_P / self.U_P**0.5 + self.V_Q / self.U_Q**0.5 ) / (self.U_P**0.5 + self.U_Q**0.5)
        
        # return test statistics 
        if signal_P * signal_Q > 0:
            return self.U_P * signal_P**2 + self.U_Q * signal_Q**2
        else:
            return max(self.U_P * signal_P**2 , self.U_Q * signal_Q**2) 
    
   
        
    def predict(self, test_X):
  
        y_predict = np.full(test_X.shape[0], self.y_hat)
        return y_predict
    
    def separate_predict(self, test_X):
        
        y_predict_P = np.full(test_X.shape[0], int(self.V_P / self.U_P > 1/2))
        y_predict_Q = np.full(test_X.shape[0], int(self.V_Q / self.U_Q > 1/2))
        return y_predict_P, y_predict_Q
    
    
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
        
    