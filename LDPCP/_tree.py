import numpy as np
from collections import deque
from copy import deepcopy

_TREE_LEAF = -1
_TREE_UNDEFINED = -2


class TreeStruct(object):
    """ Basic Binary Tree Structure.
    
    
    Parameters
    ----------
        

    
    n_features : int
        Number of dimensions.
        
    Attributes
    ----------
    node_count : int
        Log of number of nodes.
        
    left_child : list
        Log of all left children nodes.
        
    right_child : list
        Log of all right children nodes.
        
    feature : array of int
        Log of splitting dimensions.
    
    threshold : array of float 
        Log of splitting points.
        
    n_node_samples : array of int
        Log of number of points in each node.
        
    node_range : list
        Log of boundary of nodes. 
    
    """
    def __init__(self, n_features):
        
        # Base tree statistic
        self.n_features = n_features
        self.node_count = 0
        # Base inner tree struct
        self.left_child = []
        self.right_child = []
        self.feature = []
        self.threshold = []
        self.n_node_samples = []
        self.leaf_ids = []
        self.leafnode_fun = {} 
        self.allnode_fun = {}
        self.parent = []

       

        self.node_range = []
            
    def _node_append(self):
        """Add None to each logs as placeholders. 
        """
        self.left_child.append(None)
        self.right_child.append(None)
        self.feature.append(None)
        self.threshold.append(None)
        self.n_node_samples.append(None) 
        self.node_range.append(None)  
            
    def _add_node(self, parent, is_left, is_leaf, feature, threshold, n_node_samples, node_range=None):
        """Add a new node. 
        """
        
        self.parent.append(parent)
        self._node_append()
        node_id = self.node_count
        self.n_node_samples[node_id] = n_node_samples
        self.node_range[node_id] = node_range.copy()
            
        # record children status in parent nodes
        if parent != _TREE_UNDEFINED:
            if is_left:
                self.left_child[parent] = node_id
            else:
                self.right_child[parent] = node_id
        # record current node status
        if is_leaf:
            self.left_child[node_id] = _TREE_LEAF
            self.right_child[node_id] = _TREE_LEAF
            self.feature[node_id] = _TREE_UNDEFINED
            self.threshold[node_id] = _TREE_UNDEFINED
            self.leaf_ids.append(node_id)  
        else:
            # left_child and right_child will be set later
            self.feature[node_id] = feature
            self.threshold[node_id] = threshold
        self.node_count += 1
        return node_id
    
    def _node_info_to_ndarray(self):
        """Turn each logs into arrays. 
        """
        self.left_child = np.array(self.left_child, dtype=np.int32)
        self.right_child = np.array(self.right_child, dtype=np.int32)
        self.feature = np.array(self.feature, dtype=np.int32)
        self.threshold = np.array(self.threshold, dtype=np.float64)
        self.n_node_samples = np.array(self.n_node_samples, dtype=np.int32)
        self.leaf_ids = np.array(self.leaf_ids, dtype=np.int32)
        self.node_range = np.array(self.node_range, dtype=np.float64)
            
    def apply(self, X):
        """Get node ids.
        """
        return self._apply_dense(X)
    
    def _apply_dense(self, X):
        """Get node ids.
        """
        n = X.shape[0]
        result_nodeid = np.zeros(n, dtype=np.int32)
        for i in range(n):
            node_id = 0
            while self.left_child[node_id] != _TREE_LEAF:
                if X[i, self.feature[node_id]] < self.threshold[node_id]:
                    node_id = self.left_child[node_id]
                else:
                    node_id = self.right_child[node_id]
            result_nodeid[i] = node_id
        return  result_nodeid  
    
    
    
    def predict(self, X):
        """Predict for each x in X. 
        """
        node_affi = self.apply(X)
        y_predict_hat = np.zeros(X.shape[0])
        for leaf_id in self.leaf_ids:
            idx = node_affi == leaf_id
            
            y_predict_hat[idx] = self.leafnode_fun[leaf_id].predict(X[idx])
            
        return y_predict_hat
    
    def separate_predict(self, X):
        """Predict for each x in X. 
        """
        node_affi = self.apply(X)
        y_predict_hat_P = np.zeros(X.shape[0])
        y_predict_hat_Q = np.zeros(X.shape[0])
        for leaf_id in self.leaf_ids:
            idx = node_affi == leaf_id
            
            y_predict_hat_P[idx], y_predict_hat_Q[idx] = self.leafnode_fun[leaf_id].separate_predict(X[idx])
            
        return y_predict_hat_P, y_predict_hat_Q
    
    
    def predict_proba(self, X):
        """Predict for each x in X. 
        """
        node_affi = self.apply(X)
        y_predict_hat = np.zeros(X.shape[0])
        for leaf_id in self.leaf_ids:
            idx = node_affi == leaf_id
            
            y_predict_hat[idx] = self.leafnode_fun[leaf_id].predict_proba(X[idx])
            
        return y_predict_hat
    
    def separate_predict_proba(self, X):
        """Predict for each x in X. 
        """
        node_affi = self.apply(X)
        y_predict_hat_P = np.zeros(X.shape[0])
        y_predict_hat_Q = np.zeros(X.shape[0])
        for leaf_id in self.leaf_ids:
            idx = node_affi == leaf_id
            
            y_predict_hat_P[idx], y_predict_hat_Q[idx] = self.leafnode_fun[leaf_id].separate_predict_proba(X[idx])
            
        return y_predict_hat_P, y_predict_hat_Q
    
    
    
    def k_ancestor(self, node_idx, k):
        ancestor_idx = node_idx
        
        for _ in range(k):
            ancestor_idx = self.parent[ancestor_idx]
            if ancestor_idx == _TREE_UNDEFINED:
                raise ValueError("Root node")
                
        return ancestor_idx
                
    def all_descendant(self, node_idx):
        
        child_idx_list = deque()
        child_idx_list.append(node_idx)
        final_idx_list = []
        
        while len(child_idx_list) != 0:
            # dead loop
#             print(child_idx_list)
            current_idx = child_idx_list.popleft()
            if self.left_child[current_idx] != _TREE_LEAF:
                child_idx_list.append(self.left_child[current_idx]) 
            if self.right_child[current_idx] != _TREE_LEAF:
                child_idx_list.append(self.right_child[current_idx])
            if self.left_child[current_idx] == _TREE_LEAF and self.right_child[current_idx] == _TREE_LEAF:
                final_idx_list.append(current_idx)
        
        return final_idx_list
    
    
    def k_ancestor_neighbor(self, node_idx, k):
        
        ancestor_idx = self.k_ancestor(node_idx, k)
#         print(ancestor_idx)
        return self.all_descendant(ancestor_idx)
    
    
    
    
            
        


class RecursiveTreeBuilder(object):
    """ Recursively build a binary tree structure.
    
    
    Parameters
    ----------
    splitter : splitter object
        
    Estimator : estimator object
    
    epsilon : float
        Noise level.
    
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    
    min_samples_leaf : int
        The minimum number of samples required in the subnodes to split an internal node.
    
    max_depth : int
        Maximum depth of the individual regression estimators.
        
    
    """
    def __init__(self, splitter, 
                 Estimator, 
                 NodeEstimator,
                 if_prune,
                 min_samples_split, 
                 min_samples_leaf,
                 max_depth,
                 min_depth,
                 epsilon,
                 lamda,
                 lepski_ratio, 
                 ):
        
        # about splitter
        self.splitter = splitter
        # about estimator
        self.Estimator = Estimator
        self.NodeEstimator = NodeEstimator
        # about recursive splits
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.epsilon = epsilon
        self.lamda = lamda 
        self.lepski_ratio = lepski_ratio
       
        
        
    def build(self, tree, X, Y, X_range = None):
        """Grow the tree.
        """
        
        stack = deque()
        # prepare for stack [X, node_range, parent_status, left_node_status, depth]
        stack.append([X, Y, X_range, _TREE_UNDEFINED, _TREE_UNDEFINED, 0])
        while len(stack) != 0:
            dt_X, dt_Y, node_range, parent, is_left, depth = stack.popleft()
            if dt_X is None and dt_Y is None:
                n_node_samples = 0
                if depth >= self.max_depth:
                    is_leaf = True
                else: 
                    is_leaf = False
            else:
                n_node_samples = dt_X.shape[0]

                # judge whether dt should be splitted or not
                if n_node_samples <= self.min_samples_split:
                    is_leaf = True
                else:
                    n_node_unique_samples = np.unique(np.hstack([dt_X,dt_Y.reshape(-1,1)]), axis=0).shape[0]

                    # if too deep or contains too less samples, no split
                    if depth >= self.max_depth or n_node_unique_samples <= self.min_samples_split:
                        is_leaf = True
                    else:
    #                     print("At depth {}".format(depth))
                        rd_dim_vec, rd_split_vec = self.splitter(dt_X, node_range , dt_Y )
    #                     print(rd_dim_vec, rd_split_vec)
                        is_leaf = True
                        for idx_rd_dim in range(len(rd_dim_vec)):
                            rd_dim = rd_dim_vec[idx_rd_dim]
                            rd_split = rd_split_vec[idx_rd_dim]

                            if rd_split is None:
                                continue
                            elif (dt_X[:,rd_dim] >= rd_split).sum() < self.min_samples_leaf or (dt_X[:, rd_dim] < rd_split).sum() < self.min_samples_leaf:
                                continue
                            else:
                                is_leaf = False
                                break
                            

            if not is_leaf:
                node_id = tree._add_node(parent, is_left, is_leaf, rd_dim, rd_split, n_node_samples, node_range)
            else:
                node_id = tree._add_node(parent, is_left, is_leaf, None, None, n_node_samples, node_range)
  

                tree.leafnode_fun[node_id] = self.Estimator( self.max_depth,
                                                            0,
                                                            self.epsilon, 
                                                            self.lamda,
                                                            node_range,
                                                            lepski_ratio = self.lepski_ratio,
                                                            n_Q = X.shape[0]
                                                            )

                tree.leafnode_fun[node_id].fit(dt_X, dt_Y)
            
            # begin branching if the node is not leaf
            if not is_leaf:
                # update node range status
                if node_range is not None:
                    node_range_right = node_range.copy()
                    node_range_left = node_range.copy()
                    node_range_right[0, rd_dim] = rd_split
                    node_range_left[1, rd_dim] = rd_split
                else:
                    node_range_right = node_range_left = None
                    
                # push right child on stack
                if dt_X is None and dt_Y is None:
                    dt_X_right, dt_Y_right = None, None
                    dt_X_left, dt_Y_left = None, None
                else:
                    right_idx = dt_X[:,rd_dim] >= rd_split
                    dt_X_right = dt_X[right_idx]
                    dt_Y_right = dt_Y[right_idx]
                    left_idx = ~right_idx
                    dt_X_left = dt_X[left_idx]
                    dt_Y_left = dt_Y[left_idx]
                    
                stack.append([dt_X_right, dt_Y_right , node_range_right, node_id, False, depth+1])
                stack.append([dt_X_left, dt_Y_left, node_range_left, node_id, True, depth+1])
           
                
        tree._node_info_to_ndarray()
        
        
        
        
        
    def prune(self, tree, n_all):
        tree.allnode_fun = deepcopy(tree.leafnode_fun)
        flag = 0
        
        for node_idx in list(tree.leafnode_fun.keys()):
        
            current_idx = node_idx
            test_statistic_dic = {}

            # initialization 
#             print("################################### initialization idx", current_idx)
            test_statistic_dic[current_idx] = tree.allnode_fun[current_idx].test_statistic() 
            ############################################################
            # test leaf node; if pass, remain leaf node func, if not, continue
            if test_statistic_dic[current_idx] >= 1:
                tree.leafnode_fun[node_idx] = tree.allnode_fun.get(current_idx)
#                 print("################################### pass test in initial node", current_idx)
                continue

            ancestor_depth = 0
            if_terminated = 0
            # if not root node 
            while tree.parent[current_idx]  !=  _TREE_UNDEFINED:
                # to parent node
                current_idx = tree.parent[current_idx]
                ancestor_depth += 1

                # if parent node is not specified and calculated, add it to allnode_fun
                if tree.allnode_fun.get(current_idx) == None:

                    descendent_leaf_idx = tree.k_ancestor_neighbor(current_idx, 0)
                    node_fun = self.NodeEstimator(
                                             self.max_depth,
                                             ancestor_depth,
                                             self.epsilon,
                                             self.lamda,
                                             lepski_ratio = self.lepski_ratio)
                    node_fun.get_data([tree.leafnode_fun.get(idx) for idx in descendent_leaf_idx])
                    tree.allnode_fun[current_idx] = node_fun

                # add test statistic to test_statistic_dic, if parent node is already specified, this step is not necessary
#                 print("##### pruned idx", current_idx)
                test_statistic_dic[current_idx] = tree.allnode_fun.get(current_idx).test_statistic() 
                #########################################################

                # if parent node; if pass, replace leaf node func, if not, continue
                if test_statistic_dic[current_idx] >= 1:
#                     print("pass test on ancestor node with depth {}".format(ancestor_depth))
                    tree.leafnode_fun[node_idx] = tree.allnode_fun.get(current_idx)
                    if_terminated = 1
#                     print("################################### succeed in pruning with node", current_idx)
                    break
                elif ancestor_depth >= self.max_depth - self.min_depth:
#                     print("################################### reach min depth", current_idx)
                    flag = 1
                    break
                else: 
                    pass
                
            if not if_terminated:
            # return max test statistic
                
                max_statistic_idx = max(test_statistic_dic, key= lambda x: test_statistic_dic[x])
                tree.leafnode_fun[node_idx] = tree.allnode_fun.get(max_statistic_idx)
#                 print("################################### failed in pruning with node", max_statistic_idx)
#                 print("pass test on final selection")

        return flag
                
            
 
        
        
