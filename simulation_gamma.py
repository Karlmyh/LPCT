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
    
    log_file_dir = "./results/gamma/"

    np.random.seed(iterate)
    for gamma in [0.5, 0.75, 1, 1.25, 1.5, 2]:
        sample_generator = TransferDistribution(distribution_index).returnDistribution(gamma = gamma)
        n_test = 2000
        X_P, y_P, X_Q, y_Q = sample_generator.generate(n_train, n_pub)
        X_P_test, y_P_test, _, _ = sample_generator.generate(n_test, 10)


        ################################################################################################        
        method = "LDPTC-M"
        param_dict =   {"min_samples_split":[1],
                        "min_samples_leaf":[1, 5, 10, 50, 100],
                        "max_depth":[1, 2, 3, 4, 5, 6, 7, 8],
                        "lamda": [ 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 500, 750, 1000, 1250, 1500, 2000, 4000, 8000],
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
                logs= "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(gamma,
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
    num_repetitions = 100  
    num_jobs = 50        
    for epsilon in [0.5, 1, 2, 4, 8, 1000]: 
        print(epsilon)
        Parallel(n_jobs = num_jobs)(delayed(base_train)(i, epsilon, 10000, 50, 9) for i in range(num_repetitions))