import numpy as np
from ._criterion import infogain

criterion_func = {"infogain": infogain}

class PurelyRandomSplitter(object):    
    """Purely random splitter class.

    Parameters
    ----------
    random_state : int
        Random state for dimension subsampling and splitting.
    
        
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    
    Returns
    -------
    rd_dim : int in 0, ..., dim - 1
        The splitting dimension.
        
    rd_split : float
        The splitting point.

    """
    def __init__(self, random_state = None, search_number = None, threshold = None):
        self.random_state = random_state
#         np.random.seed(self.random_state)
        
    def __call__(self, X, X_range, dt_Y = None):
        n_node_samples, dim = X.shape
        
        # randomly choose dimension and split point
        rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = np.random.uniform(rddim_min, rddim_max)
        return [rd_dim], [rd_split]
    
    
class MidPointRandomSplitter(object):
    """Random mid-point splitter class.

    Parameters
    ----------
    random_state : int
        Random state for dimension subsampling and splitting.
     
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    
    Returns
    -------
    rd_dim : int in 0, ..., dim - 1
        The splitting dimension.
        
    rd_split : float
        The splitting point.

    """
    def __init__(self, random_state = None, search_number = None, threshold = None):
        self.random_state = random_state
#         np.random.seed(self.random_state)
       
        
    def __call__(self, X, X_range, dt_Y = None):
        n_node_samples, dim = X.shape
        
        # randomly choose a dimension and split at mid-point
        rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = (rddim_min+ rddim_max)/2
        return [rd_dim], [rd_split]
    
    
    
class GainReductionMidpointSplitter(object):
    """Abstract information gain based mid-point splitter class.

    Parameters
    ----------
    random_state : int
        Random state for dimension subsampling and splitting.
        
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    
    Returns
    -------
    rd_dim : int in 0, ..., dim - 1
        The splitting dimension.
        
    rd_split : float
        The splitting point.

    """
    def __init__(self, criterion, random_state = None, search_number = None, threshold = None):
        self.random_state = random_state
#         np.random.seed(self.random_state)
        self.compute_criterion_reduction = criterion_func[criterion]
        self.threshold = threshold
        
    def __call__(self, X, X_range, dt_Y):
        n_node_samples, dim = X.shape
        edge_ratio = X_range[1] - X_range[0]
        
        # sub-sample a subset of dimensions
        split_dim_vec = []
        split_point_vec = []
        criterion_vec = []

        # search for dimension with maximum criterion reduction
        for rd_dim in range(dim):
            split = ( X_range[1, rd_dim] + X_range[0, rd_dim] ) / 2
            split_dim_vec.append(rd_dim)
            split_point_vec.append(split)
            criterion_vec.append(self.compute_criterion_reduction(X, dt_Y, rd_dim, split))

            
        sorted_indices = sorted(range(len(criterion_vec)), key=lambda i: criterion_vec[i], reverse=True)
        sorted_indices = [idx for idx in sorted_indices if criterion_vec[idx] >= self.threshold]
        ratio_of_dims_totake = max(1, (len(sorted_indices) + 5 ) // 10 )
        sorted_indices = sorted_indices[0:ratio_of_dims_totake]
        sorted_mse = [criterion_vec[i] for i in sorted_indices]
        sorted_split_point = [split_point_vec[i] for i in sorted_indices]
        sorted_split_dim = [split_dim_vec[i] for i in sorted_indices]


        return sorted_split_dim, sorted_split_point
    


class InformationGainReductionMidpointSplitter(GainReductionMidpointSplitter):
    """information gain reduction mid-point splitter class.
    """
    def __init__(self, random_state = None, search_number = None, threshold = None):
        super(InformationGainReductionMidpointSplitter, self).__init__( criterion = "infogain", 
                                                   random_state = random_state, 
                                                   search_number = search_number,
                                                   threshold = threshold)
    
    
class MaxEdgeRandomSplitter(object):
    """Random max-edge splitter class.

    Parameters
    ----------
    random_state : int
        Random state for dimension subsampling and splitting.

    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    
    Returns
    -------
    rd_dim : int in 0, ..., dim - 1
        The splitting dimension.
        
    rd_split : float
        The splitting point.

    """
    def __init__(self, random_state = None, search_number = None, threshold = None):
        self.random_state = random_state
#         np.random.seed(self.random_state)
        
    def __call__(self, X, X_range ,dt_Y = None):
        n_node_samples, dim = X.shape
        
        # randomly choose among the dimensions with longest edge
        edge_ratio = X_range[1] - X_range[0]
        
        # sub-sample a subset of dimensions
        rd_dim =  np.random.choice(np.where(edge_ratio == edge_ratio.max())[0])
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = (rddim_min + rddim_max)/2
        return [rd_dim], [rd_split]
    
    
    
    
    
    
class GainReductionSplitter(object):
    """Abstract information gain based splitter class.

    Parameters
    ----------
    random_state : int
        Random state for dimension subsampling and splitting.
 
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    
    Returns
    -------
    rd_dim : int in 0, ..., dim - 1
        The splitting dimension.
        
    rd_split : float
        The splitting point.

    """
    def __init__(self, criterion, random_state = None, search_number = 10, threshold = None):
        self.random_state = random_state
#         np.random.seed(self.random_state)
        self.search_number = search_number
        self.compute_criterion_reduction = criterion_func[criterion]
        self.threshold = threshold
        
    def __call__(self, X, X_range, dt_Y):
        n_node_samples, dim = X.shape
        
        # sub-sample a subset of dimensions

        split_dim_vec = []
        split_point_vec = []
        criterion_vec = []
        
        
        # search for dimension and split point with maximum criterion reduction
        for d in range(dim):
            
            dt_X_dim_unique = np.unique(X[:,d])
            sorted_split_point = np.unique( np.quantile( dt_X_dim_unique, [(2 * i + 1)/(2 * self.search_number) for i in range(self.search_number) ] ) )
            
#             if d in [0,1]:
#                 print(sorted_split_point)
            
            split_point = None
            max_criterion_reduction = 0
            for split in sorted_split_point:
                
                criterion_reduction = self.compute_criterion_reduction(X, dt_Y, d, split)
            
                # hault if reduction is small
                if criterion_reduction > max_criterion_reduction and criterion_reduction >= self.threshold:
                    
                    max_criterion_reduction = criterion_reduction
                    split_point = split
                
            split_dim_vec.append(d)
            split_point_vec.append(split_point)
            criterion_vec.append(max_criterion_reduction)


        sorted_indices = sorted(range(len(criterion_vec)), key=lambda i: criterion_vec[i], reverse=True)
        sorted_indices = [idx for idx in sorted_indices if criterion_vec[idx] >= self.threshold]
        ratio_of_dims_totake = max(1, (len(sorted_indices) + 5 ) // 10 )
        sorted_indices = sorted_indices[0:ratio_of_dims_totake]
        sorted_mse = [criterion_vec[i] for i in sorted_indices]
        sorted_split_point = [split_point_vec[i] for i in sorted_indices]
        sorted_split_dim = [split_dim_vec[i] for i in sorted_indices]

        return sorted_split_dim, sorted_split_point
    

class InformationGainReductionSplitter(GainReductionSplitter):
    """information gain reduction splitter class.
    """
    def __init__(self, random_state = None, search_number = 10, threshold = None):
        super(InformationGainReductionSplitter, self).__init__( criterion = "infogain", 
                                                   random_state = random_state, 
                                                   search_number = search_number,
                                                   threshold = threshold)
    
    








class GainReductionMaxEdgeSplitter(object):
    """Abstract information gain based mid-point splitter class.

    Parameters
    ----------
    random_state : int
        Random state for dimension subsampling and splitting.
        
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    
    Returns
    -------
    rd_dim : int in 0, ..., dim - 1
        The splitting dimension.
        
    rd_split : float
        The splitting point.

    """
    def __init__(self, criterion, random_state = None, search_number = None, threshold = None):
        self.random_state = random_state
#         np.random.seed(self.random_state)
        self.compute_criterion_reduction = criterion_func[criterion]
        self.threshold = threshold
        
    def __call__(self, X, X_range, dt_Y):
        n_node_samples, dim = X.shape
        
      
        edge_ratio = X_range[1] - X_range[0]
        
        # sub-sample a subset of dimensions
        
        max_edges = np.where(edge_ratio == edge_ratio.max())[0]
        split_dim_vec = []
        split_point_vec = []
        criterion_vec = []
        
        # print(X_range)
        
        # search for dimension with maximum criterion reduction
        for rd_dim in max_edges:
            split = ( X_range[1, rd_dim] + X_range[0, rd_dim] ) / 2
            split_dim_vec.append(rd_dim)
            split_point_vec.append(split)
            criterion_vec.append(self.compute_criterion_reduction(X, dt_Y, rd_dim, split))
            # print(rd_dim, split)
            # print("reduction", self.compute_criterion_reduction(X, dt_Y, rd_dim, split))

            
        sorted_indices = sorted(range(len(criterion_vec)), key=lambda i: criterion_vec[i], reverse=True)
        sorted_indices = [idx for idx in sorted_indices if criterion_vec[idx] >= self.threshold]
        ratio_of_dims_totake = max(1, (len(sorted_indices) + 5 ) // 10 )
        sorted_indices = sorted_indices[0:ratio_of_dims_totake]
        sorted_mse = [criterion_vec[i] for i in sorted_indices]
        sorted_split_point = [split_point_vec[i] for i in sorted_indices]
        sorted_split_dim = [split_dim_vec[i] for i in sorted_indices]

        # print(sorted_split_dim)
        return sorted_split_dim, sorted_split_point
    


class InformationGainReductionMaxEdgeSplitter(GainReductionMaxEdgeSplitter):
    """information gain reduction mid-point splitter class.
    """
    def __init__(self, random_state = None, search_number = None, threshold = None):
        super(InformationGainReductionMaxEdgeSplitter, self).__init__( criterion = "infogain", 
                                                   random_state = random_state, 
                                                   search_number = search_number,
                                                   threshold = threshold)
