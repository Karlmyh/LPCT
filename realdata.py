import numpy as np
import os
import time
import scipy
import math
import torch
import pandas as pd
from itertools import product
import argparse
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

parser = argparse.ArgumentParser(description='Real data')
parser.add_argument('-method', type = str, default = "LPCT")
args = parser.parse_args()


data_file_dir = "./data/clean/"
data_file_name_seq = [
"anonymity",
"census",
"diabetes",
"election",
"email",
"employ",
"employee",
"jobs",
"rice",
"landcover",
"taxidata",
]
repeat_times = 50
log_file_dir = "./results/realdata/"

def base_train_LPCT_M(iterate, epsilon, dataname):

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
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 0.2, random_state = iterate)


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
        "lamda": [ 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 200, 300, 400, 500, 750, 1000],
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
                                                                time_used,
                                                                params["max_depth"],
                                                                params["min_samples_leaf"],
                                                                params["lamda"],
                                                                )
            f.writelines(logs)


def base_train_LPCT_V(iterate, epsilon, dataname):

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
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 0.2, random_state = iterate)


    # set constant
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n_test = X_test.shape[0]
    d = X_P.shape[1]

    # LPCT-V
    method = "LPCT-V"
    param_dict =   {"min_samples_split":[1],
        "min_samples_leaf":[2**(i ) for i in range(int(np.log2(n_P) - 1))],
        "max_depth":[1, 2, 3, 4, 5, 6, 7, 8,10,12,14,16],
        "lamda": [ 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 200, 300, 400, 500, 750, 1000],
        "X_Q":[X_Q],
        "y_Q": [y_Q],
        "epsilon": [epsilon],
        "splitter": ['igreduction'],
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
                                                                time_used,
                                                                params["max_depth"],
                                                                params["min_samples_leaf"],
                                                                params["lamda"],
                                                                )
            f.writelines(logs)



def base_train_LPCT_prune(iterate, epsilon, dataname):

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
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 0.2, random_state = iterate)


    # set constant
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n_test = X_test.shape[0]
    d = X_P.shape[1]

    # LPCT
    method = "LPCT-Prune"
    param_dict =   {"min_samples_split":[1],
                    "min_samples_leaf":[1, 5],
                    "if_prune": [1],
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
                                                                time_used,
                                                                0,
                                                                params["min_samples_leaf"],
                                                                0,
                                                                )
            f.writelines(logs)


def base_train_LPCT_prune_V(iterate, epsilon, dataname):

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
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 0.2, random_state = iterate)


    # set constant
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n_test = X_test.shape[0]
    d = X_P.shape[1]

    # LPCT
    method = "LPCT-Prune-V"
    param_dict =   {"min_samples_split":[1],
                    "min_samples_leaf":[1, 5],
                    "if_prune": [1],
                "X_Q":[X_Q],
                "y_Q": [y_Q],
                "epsilon": [epsilon],
                "splitter": ['igreduction'],
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
                                                                time_used,
                                                                0,
                                                                params["min_samples_leaf"],
                                                                0,
                                                                )
            f.writelines(logs)



def base_train_LPCT_original(iterate, epsilon, dataname):

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
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 0.2, random_state = iterate)


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
                                                                time_used,
                                                                params["max_depth"],
                                                                params["min_samples_leaf"],
                                                                params["lamda"],
                                                                )
            f.writelines(logs)



def base_train_PHIST(iterate, epsilon, dataname):

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
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 0.2, random_state = iterate)


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
                                                                    time_used,
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


def base_train_PGLM(iterate, epsilon, dataname):

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
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 0.2, random_state = iterate)


    # set constant
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n_test = X_test.shape[0]
    d = X_P.shape[1]

    # PGLM
    method = "LDPGLM"
    param_dict =   {"X_Q":[X_Q],
        "epsilon": [epsilon],
            }
    for param_values in product(*param_dict.values()):
        try:
            params = dict(zip(param_dict.keys(), param_values))

            time_start = time.time()
            model = LDPGLM(**params).fit(X_P, y_P)
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
                                                                    time_used,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    )
                f.writelines(logs)
        except:
            log_file_name = "{}.csv".format(method)
            log_file_path = os.path.join(log_file_dir, log_file_name)
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
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    )

def base_train_PSGD_L(iterate, epsilon, dataname):

    torch.manual_seed(iterate)
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
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 0.2, random_state = iterate)


    # set constant
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n_test = X_test.shape[0]
    d = X_P.shape[1]

   
    method = "PSGD-L"
    param_dict =   {
        "epsilon": [epsilon],
        "X_P" : [X_P],
        "y_P" : [y_P],
        "X_Q" : [X_Q],
        "y_Q" : [y_Q],
        "lr":[1e-5, 1e-3, 1e-1, 1],
            }
    for param_values in product(*param_dict.values()):
        params = dict(zip(param_dict.keys(), param_values))

        time_start = time.time()
        warm_start_model, model = train_linear(**params)
        X_test = torch.tensor(X_test, dtype = torch.double)
        y_hat = (model(X_test).detach().numpy().ravel() > 0.5).astype(int)
        eta_hat = model(X_test).detach().numpy()
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
                                                                time_used,
                                                                params["lr"],
                                                                0,
                                                                0,
                                                                )
            f.writelines(logs)

def base_train_PSGD_N(iterate, epsilon, dataname):

    torch.manual_seed(iterate)
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
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 0.2, random_state = iterate)


    # set constant
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n_test = X_test.shape[0]
    d = X_P.shape[1]

    method = "PSGD-N"
    param_dict =   {
        "epsilon": [epsilon],
        "X_P" : [X_P],
        "y_P" : [y_P],
        "X_Q" : [X_Q],
        "y_Q" : [y_Q],
        "lr":[1e-5,  1e-3,  1e-1, 1],
        "hidden_dim_ratio": [1],
            }
    for param_values in product(*param_dict.values()):
        params = dict(zip(param_dict.keys(), param_values))

        time_start = time.time()
        warm_start_model, model = train_nn(**params)
        X_test = torch.tensor(X_test, dtype = torch.double)
        eta_hat = model(X_test).detach().numpy()
        y_hat = (eta_hat.ravel() > 0.5).astype(int)
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
                                                                time_used,
                                                                params["lr"],
                                                                params["hidden_dim_ratio"],
                                                                0,
                                                                )
            f.writelines(logs)


def base_train_CT(iterate, epsilon, dataname):

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
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 0.2, random_state = iterate)


    # set constant
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n_test = X_test.shape[0]
    d = X_P.shape[1]

    # merge X_P and X_Q
    X_train = np.concatenate([X_P, X_Q], axis = 0)
    y_train = np.concatenate([y_P, y_Q], axis = 0)

    # CT
    method = "CT"
    param_dict =   {
        "max_depth":[1, 2, 3, 4, 5, 6, 7, 8,10,12,14,16],
            }
    for param_values in product(*param_dict.values()):
        params = dict(zip(param_dict.keys(), param_values))

        time_start = time.time()
        model = DecisionTreeClassifier(**params).fit(X_train, y_train)
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
                                                                time_used,
                                                                params["max_depth"],
                                                                0,
                                                                0,
                                                                )
            f.writelines(logs)


def base_train_CT_pub(iterate, epsilon, dataname):

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
    X_P, X_test, y_P, y_test = train_test_split(X, y, test_size = 0.2, random_state = iterate)


    # set constant
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n_test = X_test.shape[0]
    d = X_P.shape[1]

    # merge X_P and X_Q
    X_train = X_Q
    y_train = y_Q

    # CT
    method = "CT_pub"
    param_dict =   {
        "max_depth":[1, 2, 3, 4, 5, 6, 7, 8,10,12,14,16],
            }
    for param_values in product(*param_dict.values()):
        params = dict(zip(param_dict.keys(), param_values))

        time_start = time.time()
        model = DecisionTreeClassifier(**params).fit(X_train, y_train)
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
                                                                time_used,
                                                                params["max_depth"],
                                                                0,
                                                                0,
                                                                )
            f.writelines(logs)

    
if __name__ == "__main__":
    num_repetitions = 20
    num_jobs = 20

    
    for epsilon in [ 0.5, 1, 2, 4, 8, 1000]: 
        print(epsilon)
        for data_name in data_file_name_seq:
            print(data_name)

            if args.method == "LPCT-M":
                Parallel(n_jobs = num_jobs)(delayed(base_train_LPCT_M)(i, epsilon, data_name) for i in range(num_repetitions))
            elif args.method == "LPCT-V":
                Parallel(n_jobs = num_jobs)(delayed(base_train_LPCT_V)(i, epsilon, data_name) for i in range(num_repetitions))
            elif args.method == "LPCT-prune":
                Parallel(n_jobs = num_jobs)(delayed(base_train_LPCT_prune)(i, epsilon, data_name) for i in range(num_repetitions))
            elif args.method == "LPCT-prune-V":
                Parallel(n_jobs = num_jobs)(delayed(base_train_LPCT_prune_V)(i, epsilon, data_name) for i in range(num_repetitions))
            elif args.method == "LPCT-original":
                Parallel(n_jobs = num_jobs)(delayed(base_train_LPCT_original)(i, epsilon, data_name) for i in range(num_repetitions))
            elif args.method == "PHIST":
                num_jobs = 5
                Parallel(n_jobs = num_jobs)(delayed(base_train_PHIST)(i, epsilon, data_name) for i in range(num_repetitions))
            elif args.method == "PGLM":
                Parallel(n_jobs = num_jobs)(delayed(base_train_PGLM)(i, epsilon, data_name) for i in range(num_repetitions))
            elif args.method == "PSGD-L":
                Parallel(n_jobs = num_jobs)(delayed(base_train_PSGD_L)(i, epsilon, data_name) for i in range(num_repetitions))
            elif args.method == "PSGD-N":
                Parallel(n_jobs = num_jobs)(delayed(base_train_PSGD_N)(i, epsilon, data_name) for i in range(num_repetitions))
            elif args.method == "CT":
                Parallel(n_jobs = num_jobs)(delayed(base_train_CT)(i, epsilon, data_name) for i in range(num_repetitions))
            elif args.method == "CT_pub":
                Parallel(n_jobs = num_jobs)(delayed(base_train_CT_pub)(i, epsilon, data_name) for i in range(num_repetitions))
            else:
                raise ValueError("No such method")