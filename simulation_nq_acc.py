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


parser = argparse.ArgumentParser(description='Simulation of LPCT')


parser.add_argument('-nt', type=int, help='n train', default= 1000)
parser.add_argument('-dis', type=int, help='distribution index', default = 4)
args = parser.parse_args()






                    
def base_train(iterate, epsilon, n_train, n_pub, distribution_index):
    
    log_file_dir = "./results/nq_acc/"

    np.random.seed(iterate)
    sample_generator = TransferDistribution(distribution_index).returnDistribution()
    n_test = 2000
    X_P, y_P, X_Q, y_Q = sample_generator.generate(n_train, n_pub)
    X_P_test, y_P_test, _, _ = sample_generator.generate(n_test, 10)


    ################################################################################################        
    method = "LDPTC-M"
    param_dict =   {"min_samples_split":[1],
                    "min_samples_leaf":[1, 5, 10, 50, 100],
                    "max_depth":[1, 2, 3, 4, 5, 6, 7, 8],
                    "lamda": [ 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 2000, 4000, 8000],
                    "X_Q":[X_Q],
                    "y_Q": [y_Q],
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
                                                                 params["min_samples_leaf"],
                                                                 params["lamda"],
                                                                )
            f.writelines(logs)

    ################################################################################################        
    method = "LDPTC-M-P"
    param_dict =   {"min_samples_split":[1],
                    "min_samples_leaf":[1, 5, 10, 50, 100],
                    "max_depth":[1,2,3,4,5,6],
                    "lamda": [1],
                    "X_Q":[X_Q],
                    "y_Q": [y_Q],
                    "epsilon": [epsilon],
                    "splitter": ['igmaxedge'],
                    "estimator":["laplace"],
                      }
    for param_values in product(*param_dict.values()):
        params = dict(zip(param_dict.keys(), param_values))

        time_start = time.time()
        model = LDPTreeClassifier(**params).fit(X_P, y_P)
        y_P_hat, _ = model.separate_predict(X_P_test)
        eta_P_hat, _ = model.separate_predict_proba(X_P_test)
        accuracy = (y_P_hat == y_P_test).mean()
        bce = - log_loss(y_P_test, eta_P_hat)
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
                                                                 params["min_samples_leaf"],
                                                                 params["lamda"],
                                                                )
            f.writelines(logs)



    ################################################################################################        
    method = "LDPTC-M-Q"
    param_dict =   {"min_samples_split":[1],
                    "min_samples_leaf":[1, 5, 10, 50, 100],
                    "max_depth":[1,2,3,4,5,6],
                    "lamda": [1],
                    "X_Q":[X_Q],
                    "y_Q": [y_Q],
                    "epsilon": [epsilon],
                    "splitter": ['igmaxedge'],
                    "estimator":["laplace"],
                      }
    for param_values in product(*param_dict.values()):
        params = dict(zip(param_dict.keys(), param_values))

        time_start = time.time()
        model = LDPTreeClassifier(**params).fit(X_P, y_P)
        _, y_Q_hat = model.separate_predict(X_P_test)
        _, eta_Q_hat = model.separate_predict_proba(X_P_test)
        accuracy = (y_Q_hat == y_P_test).mean()
        bce = - log_loss(y_P_test, eta_Q_hat)
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
                                                                 params["min_samples_leaf"],
                                                                 params["lamda"],
                                                                )
            f.writelines(logs)





                    
if __name__ == "__main__":
    
    n_pub_vec = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
    num_repetitions = 300  
    num_jobs = 50        
    
    for epsilon in [0.5, 1, 2, 4, 8, 1000]: 
        print(epsilon)
        for n_pub in n_pub_vec:
            Parallel(n_jobs = num_jobs)(delayed(base_train)(i, epsilon, args.nt, n_pub, args.dis) for i in range(num_repetitions))