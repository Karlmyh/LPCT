
import numpy as np

####### all x in [0,1]^d
class JointDistribution(object): 
    def __init__(self, marginal_obj, regression_obj_P, regression_obj_Q, generation_obj, X_range = None):
        self.marginal_obj = marginal_obj
        self.regression_obj_P = regression_obj_P
        self.regression_obj_Q = regression_obj_Q
        self.generation_obj = generation_obj
        self.X_range = X_range
        
        if self.X_range is None:
            self.X_range = np.array([np.zeros(self.marginal_obj.dim),np.ones(self.marginal_obj.dim)])
        
        
    def generate(self, n_P, n_Q):
        
        X_P = self.marginal_obj.generate(n_P)
        y_P_true = self.regression_obj_P.apply(X_P)
        y_P = self.generation_obj.apply(y_P_true)
        
        X_Q = self.marginal_obj.generate(n_Q)
        y_Q_true = self.regression_obj_Q.apply(X_Q)
        y_Q = self.generation_obj.apply(y_Q_true)

        return X_P, y_P, X_Q, y_Q
    
    def generat_true(self, n_P, n_Q):
        
        X_P = self.marginal_obj.generate(n_P)
        y_P_true = self.regression_obj_P.apply(X_P)
        
        X_Q = self.marginal_obj.generate(n_Q)
        y_Q_true = self.regression_obj_Q.apply(X_Q)
        
        return X_P, y_P_true, X_Q, y_Q_true
    
    def evaluate(self, X_P = None, X_Q = None):
        
        y_P_true, y_Q_true = None, None
        
        
        
        if X_P is not None:
            check_lowerbound = (X_P - self.X_range[0] >= 0).all(axis = 1)
            check_upperbound = (X_P - self.X_range[1] <= 0).all(axis = 1)
            is_inboundary = check_lowerbound * check_upperbound
            
            y_P_true = self.regression_obj_P.apply(X_P)
            y_P_true[np.logical_not(is_inboundary)] = 0.5
            
        if X_Q is not None:
            check_lowerbound = (X_Q - self.X_range[0] >= 0).all(axis = 1)
            check_upperbound = (X_Q - self.X_range[1] <= 0).all(axis = 1)
            is_inboundary = check_lowerbound * check_upperbound
            
            y_Q_true = self.regression_obj_P.apply(X_Q)
            y_Q_true[np.logical_not(is_inboundary)] = 0.5

        return y_P_true, y_Q_true
    
    def label(self, X_P = None, X_Q = None):
        
        y_P, y_Q = None, None
        
        if X_P is not None:
            y_P_true = self.regression_obj_P.apply(X_P)
            y_P = self.generation_obj.apply(y_P_true)
        if X_Q is not None:
            y_Q_true = self.regression_obj_Q.apply(X_Q)
            y_Q = self.generation_obj.apply(y_Q_true)

        return y_P, y_Q
        