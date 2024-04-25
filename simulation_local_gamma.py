import numpy as np
import os
import time
import scipy
import math
import pandas as pd
from itertools import product
import argparse
from joblib import Parallel, delayed

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss

from distribution import TransferDistribution
from LDPCP import LDPTreeClassifier


                    
def base_train(iterate, epsilon, n_train, n_pub, distribution_index):
    
    log_file_dir = "./results/local-gamma/"

    np.random.seed(iterate)
    n_test = 2000
    sample_generator = TransferDistribution(distribution_index).returnDistribution()
    X_P, y_P, X_Q, y_Q = sample_generator.generate(n_train, n_pub)
    X_P_test, y_P_test, _, _ = sample_generator.generate(n_test, 10)

    
    
    ################################################################################################        
    method = "LDPTC-M"
    param_dict =   {"min_samples_split":[0],
                    "min_samples_leaf":[0],
                    "if_prune": [0],
                    "X_Q":[X_Q],
                    "y_Q": [y_Q],
                    "max_depth": [1,2,3,4,5,6,7,8],
                    "lamda": [0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 2000],
                    "epsilon": [epsilon],
                    "splitter": ['igmaxedge'],
                    "estimator":["laplace"],
                      }
    for param_values in product(*param_dict.values()):
        params = dict(zip(param_dict.keys(), param_values))
       

        time_start = time.time()
        model = LDPTreeClassifier(**params).fit(X_P, y_P)
        y_hat = model.predict(X_P_test)
        eta_hat = model.predict_proba(X_P_test)
        accuracy = (y_hat == y_P_test).mean()
        bce = - log_loss(y_P_test, eta_hat)
        time_end = time.time()
        time_used = time_end - time_start

        log_file_name = "{}.csv".format(method)
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(distribution_index,
                                                                 method,
                                                                 iterate,
                                                                 epsilon,
                                                                 n_train,
                                                                 n_pub, 
                                                                 accuracy,
                                                                 bce,
                                                                 time_used,
                                                                 params["max_depth"],
                                                                 0,
                                                                 params["lamda"],
                                                                )
            f.writelines(logs)


    ################################################################################################        
    method = "LDPTC-M-prune"
    param_dict =   {"min_samples_split":[0],
                    "min_samples_leaf":[0],
                    "if_prune": [1],
                    "X_Q":[X_Q],
                    "y_Q": [y_Q],
                    "epsilon": [epsilon],
                    "splitter": ['igmaxedge'],
                    "estimator":["laplace"],
                      }
    for param_values in product(*param_dict.values()):
        params = dict(zip(param_dict.keys(), param_values))
        params["max_depth"] = np.floor(np.log2(n_train * epsilon**2 + n_pub**3) / 3)
        params["min_depth"] = np.floor(np.log2(n_train * epsilon**2 + 1) / 3)
       

        time_start = time.time()
        model = LDPTreeClassifier(**params).fit(X_P, y_P)
        y_hat = model.predict(X_P_test)
        eta_hat = model.predict_proba(X_P_test)
        accuracy = (y_hat == y_P_test).mean()
        bce = - log_loss(y_P_test, eta_hat)
        time_end = time.time()
        time_used = time_end - time_start

        log_file_name = "{}.csv".format(method)
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(distribution_index,
                                                                 method,
                                                                 iterate,
                                                                 epsilon,
                                                                 n_train,
                                                                 n_pub, 
                                                                 accuracy,
                                                                 bce,
                                                                 time_used,
                                                                 params["max_depth"],
                                                                 params["min_depth"],
                                                                 0,
                                                                )
            f.writelines(logs)


                    
if __name__ == "__main__":
    
    n_train_vec = [10000]
    num_repetitions = 200
    num_jobs = 67   
    
    n_pub_vec = [50, 100, 200]
    for epsilon in [0.001, 0.5, 1, 2, 4, 8, 1000]: 
        print(epsilon)
        for n_train in n_train_vec:
            for n_pub in n_pub_vec:
                print(n_train, n_pub)
                Parallel(n_jobs = num_jobs)(delayed(base_train)(i, epsilon, n_train, n_pub, 8) for i in range(num_repetitions))
                    
              