import numpy as np
import os
import time
import scipy
import math
import pandas as pd
from itertools import product
from joblib import Parallel, delayed

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier

from LDPCP import LDPTreeClassifier
from comparison.PHIST import LDPHistogram
from comparison.PSGD import train_linear, train_nn
from comparison.PGLM import LDPGLM




data_file_dir = "./data/clean/"
data_file_name_seq = [
"employee",
# "diabetes",
]
log_file_dir = "./results/npnq/"

def base_train_LPCT_M(iterate, epsilon, dataname, delta):

    pub_file_name = "{}_pub.csv".format(dataname)
    path = os.path.join(data_file_dir, pub_file_name)
    data = pd.read_csv(path, header=None)
    data = np.array(data, dtype = "float")
    X_Q = data[:200,1:]
    y_Q = data[:200,0]

    scalar = MinMaxScaler()
    X_Q = scalar.fit_transform(X_Q)

    pri_file_name = "{}_pri.csv".format(dataname)
    path = os.path.join(data_file_dir, pri_file_name)
    data = pd.read_csv(path, header=None)
    data = np.array(data, dtype = "float")
    X = data[:,1:]
    y = data[:,0]
    X = scalar.transform(X)
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 1 - 0.8 * delta, random_state = iterate)


    # set constant
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n_test = X_test.shape[0]
    d = X_P.shape[1]

    # LPCT-M
    method = "LPCT-M"
    param_dict =   {"min_samples_split":[1],
        "min_samples_leaf":[2**(i ) for i in range(int(np.log2(n_P) - 1))],
        "max_depth":[1, 2, 3, 4, 5, 6, 7, 8,10,12,14,16],
        "lamda": [ 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 200],
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
        y_hat = model.predict(X_test)
        eta_hat = model.predict_proba(X_test)
        accuracy = (y_hat == y_test).mean()
        bce = - log_loss(y_test, eta_hat)
        time_end = time.time()
        time_used = time_end - time_start

        log_file_name = "{}.csv".format(method)
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{},{:6f},{:6f},{:6f},{},{},{}\n".format(dataname,
                                                                method,
                                                                iterate,
                                                                epsilon,
                                                                n_P,
                                                                n_Q, 
                                                                accuracy,
                                                                bce,
                                                                delta,
                                                                params["max_depth"],
                                                                params["min_samples_leaf"],
                                                                params["lamda"],
                                                                )
            f.writelines(logs)



def base_train_LPCT_original(iterate, epsilon, dataname, delta):

    pub_file_name = "{}_pub.csv".format(dataname)
    path = os.path.join(data_file_dir, pub_file_name)
    data = pd.read_csv(path, header=None)
    data = np.array(data, dtype = "float")
    X_Q = data[:200,1:]
    y_Q = data[:200,0]

    scalar = MinMaxScaler()
    X_Q = scalar.fit_transform(X_Q)

    pri_file_name = "{}_pri.csv".format(dataname)
    path = os.path.join(data_file_dir, pri_file_name)
    data = pd.read_csv(path, header=None)
    data = np.array(data, dtype = "float")
    X = data[:,1:]
    y = data[:,0]
    X = scalar.transform(X)
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 1 - 0.8 * delta, random_state = iterate)


    # set constant
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n_test = X_test.shape[0]
    d = X_P.shape[1]

    # LPCT-original
    method = "LPCT-original"
    param_dict =   {"min_samples_split":[1],
        "min_samples_leaf":[2**(i ) for i in range(int(np.log2(n_P) - 1))],
        "max_depth":[1, 2, 3, 4, 5, 6, 7, 8,10,12,14,16],
        "lamda": [ 0],
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
        y_hat = model.predict(X_test)
        eta_hat = model.predict_proba(X_test)
        accuracy = (y_hat == y_test).mean()
        bce = - log_loss(y_test, eta_hat)
        time_end = time.time()
        time_used = time_end - time_start

        log_file_name = "{}.csv".format(method)
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{},{:6f},{:6f},{:6f},{},{},{}\n".format(dataname,
                                                                method,
                                                                iterate,
                                                                epsilon,
                                                                n_P,
                                                                n_Q, 
                                                                accuracy,
                                                                bce,
                                                                delta,
                                                                params["max_depth"],
                                                                params["min_samples_leaf"],
                                                                params["lamda"],
                                                                )
            f.writelines(logs)



def base_train_PHIST(iterate, epsilon, dataname, delta):

    pub_file_name = "{}_pub.csv".format(dataname)
    path = os.path.join(data_file_dir, pub_file_name)
    data = pd.read_csv(path, header=None)
    data = np.array(data, dtype = "float")
    X_Q = data[:,1:]
    y_Q = data[:,0]

    scalar = MinMaxScaler()
    X_Q = scalar.fit_transform(X_Q)

    pri_file_name = "{}_pri.csv".format(dataname)
    path = os.path.join(data_file_dir, pri_file_name)
    data = pd.read_csv(path, header=None)
    data = np.array(data, dtype = "float")
    X = data[:,1:]
    y = data[:,0]
    X = scalar.transform(X)
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 1 - 0.8 * delta, random_state = iterate)


    # set constant
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n_test = X_test.shape[0]
    d = X_P.shape[1]

    # PHIST
    method = "LDPHIST"
    param_dict =   {"num_cell":[1,2,3,4,5,6],
        "epsilon": [epsilon],
            }
    for param_values in product(*param_dict.values()):

        params = dict(zip(param_dict.keys(), param_values))

        if d * np.log(params["num_cell"]) <= 30 * np.log(2) and d < 32:
                
            time_start = time.time()
            model = LDPHistogram(**params).fit(X_P, y_P)
            y_hat = model.predict(X_test)
            eta_hat = model.predict_proba(X_test)
            accuracy = (y_hat == y_test).mean()
            bce = - log_loss(y_test, eta_hat)
            time_end = time.time()
            time_used = time_end - time_start

            log_file_name = "{}.csv".format(method)
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs= "{},{},{},{},{},{},{:6f},{:6f},{:6f},{},{},{}\n".format(dataname,
                                                                    method,
                                                                    iterate,
                                                                    epsilon,
                                                                    n_P,
                                                                    n_Q, 
                                                                    accuracy,
                                                                    bce,
                                                                    delta,
                                                                    params["num_cell"],
                                                                    0,
                                                                    0,
                                                                    )
                f.writelines(logs)
        else:
            log_file_name = "{}.csv".format(method)
            log_file_path = os.path.join(log_file_dir, log_file_name)
            print(log_file_path)
            with open(log_file_path, "a") as f:
                logs= "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(dataname,
                                                                    method,
                                                                    iterate,
                                                                    epsilon,
                                                                    n_P,
                                                                    n_Q, 
                                                                    np.nan,
                                                                    np.nan,
                                                                    np.nan,
                                                                    params["num_cell"],
                                                                    0,
                                                                    0,
                                                                    )
                f.writelines(logs)
    
if __name__ == "__main__":
    num_repetitions = 50  
    num_jobs = 25   

    
    for epsilon in [ 0.5, 1, 2, 4, 8, 1000]: 
        print(epsilon)
        for data_name in data_file_name_seq:
            print(data_name)

            for delta in [0.2, 0.4, 0.6, 0.8, 1]:
                print(delta)
                Parallel(n_jobs = num_jobs)(delayed(base_train_LPCT_M)(i, epsilon, data_name, delta) for i in range(num_repetitions))
                Parallel(n_jobs = num_jobs)(delayed(base_train_LPCT_original)(i, epsilon, data_name, delta) for i in range(num_repetitions))
                num_jobs = 5
                Parallel(n_jobs = num_jobs)(delayed(base_train_PHIST)(i, epsilon, data_name, delta) for i in range(num_repetitions))
                num_jobs = 25
        