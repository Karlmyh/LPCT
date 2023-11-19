import numpy as np
from sklearn.metrics import mean_squared_error as MSE
import itertools
from scipy.stats import laplace

from time import time
from sklearn.metrics import log_loss






class LDPHistogram(object):
    """ Local differential privacy histogram estimator.
    
    
    Parameters
    ----------
    num_cell : int
        Number of splits along each dimension.
        
    epsilon : float
        Scale of laplace random variable.
        
        
    Attributes
    ----------
    
    """
    def __init__(self, 
                 num_cell = 2, 
                 epsilon = 1,
                 min_samples_cell = 2,
                ):
        
        self.num_cell = num_cell
        self.epsilon = epsilon
        self.min_samples_cell = min_samples_cell
        
        
             
    def fit(self, X, Y, X_range = "unit"):
        
        self.n_samples, self.dim = X.shape
        self.X_range = np.array([np.zeros(self.dim),np.ones(self.dim)])
        
        # set noise level
        self.noise_level_Z = 4  / self.epsilon
        self.noise_level_W = 4  / self.epsilon
        
        # check the boundary
        if X_range == "unit":
            X_range = np.array([np.zeros(self.dim),np.ones(self.dim)])
        if X_range is None:
            X_range = np.zeros(shape = (2, self.dim))
            X_range[0] = X.min(axis = 0) - 0.01 * (X.max(axis = 0) - X.min(axis = 0))
            X_range[1] = X.max(axis = 0) + 0.01 * (X.max(axis = 0) - X.min(axis = 0))
        self.X_range = X_range



        # compute the edge of bins for each dimension
        self.bin_edges = []
        for i in range(self.dim):
            self.bin_edges.append(np.linspace(self.X_range[0,i], self.X_range[1,i], self.num_cell+1))
            
            
        # the storage of sum of numerator and denominator
        Z = np.zeros((self.num_cell,) * self.dim )
        W = np.zeros((self.num_cell,) * self.dim )
        
        # identify cell
        cell_idx = tuple(np.digitize(X[:,k], self.bin_edges[k]) for k in range(self.dim))
        
        
        #time_end = time()
        #print("identify cell time : {}".format(time_end - time_start))
        #time_start = time()
        
        # get data
        np.random.seed(1)
        for i in range(self.n_samples):
            data_idx = np.array([cell_idx[k][i] for k in range(self.dim)]) - 1
            data_idx[data_idx > self.num_cell - 1 ] = self.num_cell - 1
            data_idx = tuple(data_idx)
        
            Z[data_idx] += Y.ravel()[i]
            W[data_idx] += 1
        
        Z += laplace.rvs(size = (self.num_cell,) * self.dim, scale = np.sqrt(self.n_samples) * self.noise_level_Z)
        W += laplace.rvs(size = (self.num_cell,) * self.dim, scale = np.sqrt(self.n_samples) * self.noise_level_W)

        
        #time_end = time()
        #print("get data time : {}".format(time_end - time_start))
        #time_start = time()
        
        
        # precomputing
        self.y_hat = Z / W
        
        
        
        self.y_hat[W < self.min_samples_cell] = 0 
        
 
        #time_end = time()
        #print("compute y hat time : {}".format(time_end - time_start))

        
        return self
        
   
    
        
    
    def predict_proba(self, X):
        
        y_hat = np.zeros(X.shape[0])
        
        # identify cell
        time_start = time()
        
        lower_check = (X >= self.X_range[0, :]).all(axis = 1 )
        upper_check = (X <= self.X_range[1, :]).all(axis = 1 )
        in_idx = lower_check * upper_check
        
        
        X_in = X[in_idx]
        
        cell_idx = tuple(np.digitize(X_in[:,k], self.bin_edges[k]) for k in range(self.dim))

      
        
        # query precomputed prediction value
        for i in range(X_in.shape[0]):
            data_idx = np.array([cell_idx[k][i] for k in range(self.dim)]) - 1
            data_idx[data_idx > self.num_cell - 1 ] = self.num_cell - 1
       
            data_idx = tuple(data_idx)
            if in_idx[i] == 1:
                y_hat[i] = self.y_hat[data_idx]
            
        
        return y_hat
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
    

        
    def score(self, X, y):
        """Reture the score, i.e. logistic.
        """
        return - log_loss(y,self.predict(X))
    


