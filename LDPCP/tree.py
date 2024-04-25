import numpy as np

from ._tree import TreeStruct, RecursiveTreeBuilder
from ._splitter import PurelyRandomSplitter, MidPointRandomSplitter, MaxEdgeRandomSplitter, InformationGainReductionSplitter, InformationGainReductionMidpointSplitter, InformationGainReductionMaxEdgeSplitter

from ._estimator import ClassificationEstimator, AncestorNodePruningEstimator
from sklearn.metrics import log_loss


SPLITTERS = {"purely": PurelyRandomSplitter,
             "midpoint": MidPointRandomSplitter, 
             "maxedge": MaxEdgeRandomSplitter, 
             "igreduction": InformationGainReductionSplitter,
             "igmaxedge": InformationGainReductionMaxEdgeSplitter,
             "igmidpoint":InformationGainReductionMidpointSplitter,
             }

ESTIMATORS = {"laplace": ClassificationEstimator
                }

NODEESTIMATORS = {"RSS": AncestorNodePruningEstimator
                }




class BaseRecursiveTree(object):
    """ Abstact Recursive Tree Structure.
    
    
    Parameters
    ----------
    splitter : splitter keyword in SPLITTERS
        Splitting scheme
        
    noise_level : float
        Scale of laplace random variable.
        
   
        
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    
    min_samples_leaf : int
        The minimum number of samples required in the subnodes to split an internal node.
    
    max_depth : int
        Maximum depth of the individual regression estimators.
        
    random_state : int
        Random state for building the tree.
        
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    Attributes
    ----------
    n_samples : int
        Number of samples.
    
    dim : int
        Dimension of covariant.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the support, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
        
    tree_ : binary tree object defined in _tree.py
    
    """
    def __init__(self, 
                 splitter = None, 
                 epsilon = None,
                 if_prune = None,
                 min_samples_split = None,
                 min_samples_leaf = None,
                 max_depth = None, 
                 min_depth = None,
                 random_state = None,
                 search_number = None,
                 threshold = None,
                 estimator = None,
                 node_estimator = None,
                 lamda = None,
                 lepski_ratio = None,
                ):
        
        self.splitter = splitter
        self.epsilon = epsilon
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.random_state = random_state
        self.search_number = search_number
        self.threshold = threshold
        self.estimator = estimator
        self.node_estimator = node_estimator
        self.lamda = lamda
        self.if_prune = if_prune
        self.lepski_ratio = lepski_ratio
        
        if self.max_depth < self.min_depth:
            self.min_depth = self.max_depth


        
             
    def _fit(self, X, Y, range_x = "unit"):
        
        
        self.n_samples, self.dim = X.shape
        
        # check the boundary
        if range_x in ["unit"]:
            X_range = np.array([np.zeros(self.dim),np.ones(self.dim)])
        if range_x in ['auto']:
            X_range = np.zeros(shape = (2, self.dim))
            X_range[0] = X.min(axis = 0) - 0.01 * (X.max(axis = 0) - X.min(axis = 0))
            X_range[1] = X.max(axis = 0) + 0.01 * (X.max(axis = 0) - X.min(axis = 0))
        self.X_range = X_range





        # begin
        splitter = SPLITTERS[self.splitter](self.random_state, self.search_number, self.threshold)
        
        Estimator = ESTIMATORS[self.estimator]
        NodeEstimator = NODEESTIMATORS[self.node_estimator]
        
        
        
        # initiate a tree structure
        self.tree_ = TreeStruct(self.dim)
  
        # recursively build the tree
        self.builder = RecursiveTreeBuilder(splitter, 
                                       Estimator, 
                                       NodeEstimator,
                                       self.if_prune,
                                       self.min_samples_split, 
                                       self.min_samples_leaf,
                                       self.max_depth, 
                                       self.min_depth,
                                       self.epsilon,
                                       self.lamda,
                                       self.lepski_ratio
                                       )
      
        self.builder.build(self.tree_, X, Y, X_range)
        
      
        return self
    
    def pruning(self, n_all):
        pr = self.builder.prune(self.tree_, n_all)
        return self
   
        
        
    def apply(self, X):
        """Reture the belonging cell ids. 
        """
        return self.tree_.apply(X)
    
    
    def get_node_idx(self,X):
        """Reture the belonging cell ids. 
        """
        return self.apply(X)
    
    def get_node(self,X):
        """Reture the belonging node. 
        """
        return [self.tree_.leafnode_fun[i] for i in self.get_node_idx(X)]
    
    def get_all_node(self):
        """Reture all nodes. 
        """
        return list(self.tree_.leafnode_fun.values())
    
    def get_all_node_idx(self):
        """Reture all node indexes. 
        """
        return list(self.tree_.leafnode_fun.keys())
    
    
    def k_ancestor_neighbor_idx(self, node_idx, k):
        return self.tree_.k_ancestor_neighbor(node_idx, k)
    
    def k_ancestor_neighbor_nodes(self, node_idx, k):
        return [self.tree_.leafnode_fun.get(idx) for idx in self.k_ancestor_neighbor_idx(node_idx, k)]
        
    
    def predict(self, X):
        
        y_hat = self.tree_.predict(X)
        
        # check boundary
        check_lowerbound = (X - self.X_range[0] >= 0).all(axis = 1)
        check_upperbound = (X - self.X_range[1] <= 0).all(axis = 1)
        is_inboundary = check_lowerbound * check_upperbound
        # assign 0 to points outside the boundary
        y_hat[np.logical_not(is_inboundary)] = 0
        return y_hat
    
    def separate_predict(self, X):
        
        y_hat_P, y_hat_Q = self.tree_.separate_predict(X)
        
        # check boundary
        check_lowerbound = (X - self.X_range[0] >= 0).all(axis = 1)
        check_upperbound = (X - self.X_range[1] <= 0).all(axis = 1)
        is_inboundary = check_lowerbound * check_upperbound
        # assign 0 to points outside the boundary
        y_hat_P[np.logical_not(is_inboundary)] = 0
        y_hat_Q[np.logical_not(is_inboundary)] = 0
        return y_hat_P, y_hat_Q
    
    def predict_proba(self, X):
        
        y_proba = self.tree_.predict_proba(X)
        
        # check boundary
        check_lowerbound = (X - self.X_range[0] >= 0).all(axis = 1)
        check_upperbound = (X - self.X_range[1] <= 0).all(axis = 1)
        is_inboundary = check_lowerbound * check_upperbound
        # assign 0 to points outside the boundary
        y_proba[np.logical_not(is_inboundary)] = 0
        return y_proba
    
    def separate_predict_proba(self, X):
        
        y_proba_P, y_proba_Q = self.tree_.separate_predict_proba(X)
        
        # check boundary
        check_lowerbound = (X - self.X_range[0] >= 0).all(axis = 1)
        check_upperbound = (X - self.X_range[1] <= 0).all(axis = 1)
        is_inboundary = check_lowerbound * check_upperbound
        # assign 0 to points outside the boundary
        y_proba_P[np.logical_not(is_inboundary)] = 0
        y_proba_Q[np.logical_not(is_inboundary)] = 0
        return y_proba_P, y_proba_Q
    





class LDPTreeClassifier(BaseRecursiveTree):
    """Locally private classification tree with public data. 
    """
    def __init__(self, splitter = "igmaxedge", 
                 epsilon = 1,
                 if_prune = 0,
                 X_Q = None,
                 y_Q = None,
                 min_samples_split = 5, 
                 min_samples_leaf = 2,
                 min_depth = 1,
                 max_depth = 2, 
                 random_state = 666,
                 search_number = 10,
                 threshold = 0,
                 estimator = "laplace",
                 node_estimator = "RSS", 
                 lamda = None,
                 range_x = "unit",
                 lepski_ratio = 1,
                 ):
        super(LDPTreeClassifier, self).__init__(splitter = splitter,
                                             epsilon = epsilon, 
                                             if_prune = if_prune, 
                                             min_samples_split = min_samples_split,
                                             min_samples_leaf = min_samples_leaf,
                                             max_depth = max_depth, 
                                             min_depth = min_depth,
                                             random_state = random_state,
                                             search_number = search_number,
                                             threshold = threshold,
                                              estimator = estimator,
                                              node_estimator = node_estimator,
                                               lamda = lamda, 
                                                lepski_ratio = lepski_ratio,
                                              )

        self.X_Q = X_Q
        self.y_Q = y_Q
        self.range_x = range_x
        
        
    def get_partition(self, X, y, range_x = "unit"):
        """Fit the tree with public data.
        """
        
        return self._fit( X, y, range_x = range_x)
        
    
    
    def attribute_data(self, X, y):
        """Fit the estimator with private data.
        """
        test_idx = self.apply(X)
        for node_idx in self.get_all_node_idx():
           
            in_idx  = test_idx == node_idx
            

            self.tree_.leafnode_fun[node_idx].get_data(y, in_idx)
            if not self.if_prune:
                # if not pruning, update the predicted value
                self.tree_.leafnode_fun[node_idx].test_statistic()
            
    
    
    def fit(self, X, y):

        if self.if_prune:
            self.epsilon = self.epsilon / 2
            self.get_partition(self.X_Q, self.y_Q, self.range_x)
            self.attribute_data(X,y)
            flag = self.pruning(self.y_Q.shape[0] + y.shape[0])

            if flag:
                self.max_depth = self.min_depth
                self.get_partition(self.X_Q, self.y_Q, self.range_x)
                self.attribute_data(X,y)
        else:
            self.get_partition(X, y, self.range_x)
            self.attribute_data(X,y)


       
        return self
    
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in ['min_samples_split',"min_samples_leaf", "max_depth", "min_depth",
                    "splitter", "epsilon", "if_prune", "X_Q", "y_Q",
                    "search_number", "threshold", "estimator",
                   "range_x"]:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    
    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self
    
    


        
    def score(self, X, y):
        """Reture the classification score, i.e. bce loss.
        """
        return - log_loss(y,self.predict(X))
    
    

