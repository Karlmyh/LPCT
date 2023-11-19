from distribution import TransferDistribution
import numpy as np
from LDPCP import LDPTreeClassifier

def test_transferdistribution():
    distribution = TransferDistribution(1).returnDistribution()
    
    X_P, y_P, X_Q, y_Q = distribution.generate(9000, 9000)
    X_P_test, y_P_test, _, _ = distribution.generate(5000, 1000)
    
    np.random.seed(2)
    model = LDPTreeClassifier(splitter = "igmaxedge", 
                                     max_depth = 4, 
                                     X_Q = X_Q, 
                                     y_Q = y_Q,
                                     epsilon = 4,
                                     if_prune = 0,
                                     min_samples_split = 10,
                                     min_samples_leaf = 10,).fit(X_P, y_P)

    acc = (model.predict(X_P_test) == y_P_test).mean()
    score = - model.score(X_P_test, y_P_test)
    
    y_P_true, _ = distribution.evaluate(X_P = X_P_test)
    bayes_risk = np.stack([y_P_true, 1 - y_P_true] ).max(axis = 0).mean()
    
    
    pre_P, pre_Q = model.separate_predict(X_P_test)
    acc_P = (pre_P == y_P_test).mean()
    acc_Q = (pre_Q == y_P_test).mean()
    
    
    print(acc)
    print(acc_P)
    print(acc_Q)
    print(score)
    print(bayes_risk)
    
    
    np.random.seed(2)
    model = LDPTreeClassifier(splitter = "igmaxedge", 
                                     max_depth = 4, 
                                     X_Q = X_Q, 
                                     y_Q = y_Q,
                                     epsilon = 4,
                                     if_prune = 1,
                                     min_samples_split = 10,
                                     min_samples_leaf = 10,).fit(X_P, y_P)

    acc = (model.predict(X_P_test) == y_P_test).mean()
    score = - model.score(X_P_test, y_P_test)
    
    y_P_true, _ = distribution.evaluate(X_P = X_P_test)
    bayes_risk = np.stack([y_P_true, 1 - y_P_true] ).max(axis = 0).mean()
    
    
    pre_P, pre_Q = model.separate_predict(X_P_test)
    acc_P = (pre_P == y_P_test).mean()
    acc_Q = (pre_Q == y_P_test).mean()
    
    
    print(acc)
    print(acc_P)
    print(acc_Q)
    print(score)
    print(bayes_risk)
    
