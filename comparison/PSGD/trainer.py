from .model import LinearClassifier, SingleLayerNN
from .optimizer import PrivateUnitSGD

import numpy as np
import copy 
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from torch.nn import BCELoss
from torch.optim import SGD


def train_linear(epsilon, X_P, X_Q, y_P, y_Q, lr):
    """
    Train a linear classifier using the private - public SGD optimizer.
    """
    if type(X_P) is np.ndarray:
        X_P = torch.from_numpy(X_P)
        X_Q = torch.from_numpy(X_Q)
        y_P = torch.from_numpy(y_P)
        y_Q = torch.from_numpy(y_Q)

    C = 1
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n = n_P + n_Q
    d = X_P.shape[1]
    model = LinearClassifier(d)
    priv_optimizer = PrivateUnitSGD(model.parameters(), lr = lr, C = C, epsilon = epsilon)
    pub_optimizer = SGD(model.parameters(), lr = lr)
    criterion = BCELoss()

    # warm start
    for i in range(100):
        pub_optimizer.zero_grad()
        output = model(X_Q)
        loss = criterion(output, y_Q.reshape(-1,1))
        loss.backward()
        pub_optimizer.step()
        # if i % 10 == 0:
        #     print('warm start: ', i, loss.item())

    warm_start_model = copy.deepcopy(model)

    indexes = torch.randperm(n)
    for idx_i, i in enumerate(indexes):
        if i < n_P:
            x = X_P[i, :]
            y = y_P[i].reshape(1)
            priv_optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            priv_optimizer.step()
        else:
            x = X_Q[i-n_P, :]
            y = y_Q[i-n_P].reshape(1)
            pub_optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            pub_optimizer.step()
        # if idx_i % (n // 10) == 0:
        #     print('epoch: ', idx_i, i, loss.item())

    return warm_start_model, model


def train_nn(epsilon, X_P, X_Q, y_P, y_Q, lr, hidden_dim_ratio):
    """
    Train a linear classifier using the private - public SGD optimizer.
    """
    if type(X_P) is np.ndarray:
        X_P = torch.from_numpy(X_P)
        X_Q = torch.from_numpy(X_Q)
        y_P = torch.from_numpy(y_P)
        y_Q = torch.from_numpy(y_Q)

    C = 1
    n_P = X_P.shape[0]
    n_Q = X_Q.shape[0]
    n = n_P + n_Q
    d = X_P.shape[1]
    model = SingleLayerNN(d, int(hidden_dim_ratio * d))
    priv_optimizer = PrivateUnitSGD(model.parameters(), lr = lr, C = C, epsilon = epsilon)
    pub_optimizer = SGD(model.parameters(), lr = lr)
    criterion = BCELoss()

    # warm start
    for i in range(100):
        pub_optimizer.zero_grad()
        output = model(X_Q)
        loss = criterion(output, y_Q.reshape(-1,1))
        loss.backward()
        pub_optimizer.step()
        # if i % 20 == 0:
        #     print('warm start: ', i, loss.item())

    warm_start_model = copy.deepcopy(model)


    indexes = torch.randperm(n)
    for idx_i, i in enumerate(indexes):
        if i < n_P:
            x = X_P[i, :]
            y = y_P[i].reshape(1)
            priv_optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            priv_optimizer.step()
        else:
            x = X_Q[i-n_P, :]
            y = y_Q[i-n_P].reshape(1)
            pub_optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            pub_optimizer.step()
        # if idx_i % (n // 10) == 0:
        #     print('epoch: ', idx_i, i, loss.item())

    return warm_start_model, model