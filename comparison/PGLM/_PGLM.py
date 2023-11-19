import numpy as np
import math



def sigmoid(X):
    return .5 * (1 + np.tanh(.5 * X))


def phi(X):
    return np.log( 1 + np.exp(X) )

def phi1(X):
    return sigmoid(X)

def phi2(X):
    return sigmoid(X) * (1 - sigmoid(X))

def phi3(X):
    return sigmoid(X) * (1 - sigmoid(X)) * (1 - 2 * sigmoid(X))

class LDPGLM(object):
    '''
    The private generalized linear model.
    See https://shao3wangdi.github.io/papers/ALT2021_GLM.pdf

    Parameters
    ----------
    epsilon : float
        The privacy budget.
    delta : float
        The delta privacy parameter.
    X_Q : (n, d) array
        The unlabeled public dataset.
    
    Attributes
    ----------
    d : int
        The dimension of the data.
    r : float.
        The diameter of the data.
    '''
    def __init__(self, 
                 epsilon = 2, 
                 delta = None,
                 X_Q = None,
                 c = 1,
                ):
        self.epsilon = epsilon
        self.delta = delta
        self.n_Q = X_Q.shape[0]
        self.X_Q = np.hstack([np.ones(self.n_Q).reshape(-1,1), X_Q])
        self.d = self.X_Q.shape[1]
        # default [0,1]^d
        self.r = self.X_Q.shape[1]
        
        self.c = c

    def fit(self, X_P, y_P):
        self.n_P = X_P.shape[0]
        eX_P = np.hstack([np.ones(self.n_P).reshape(-1,1), X_P])
        if self.delta is None:
            self.delta = 1 / self.n_P**2

        # compute variance scale of n_P noises
        var_scale = 32 * self.r**4 * np.log(2.5 / self.delta) / self.epsilon**2 * self.n_P**0.5

        unit_gaussian_noise = np.random.normal(0, 1, size = (self.d, self.d))
        unit_gaussian_noise = (unit_gaussian_noise.T + unit_gaussian_noise) / np.sqrt(2)
        XX_hat = eX_P.T @ eX_P + unit_gaussian_noise * np.sqrt(var_scale)


        # privatize XTy
        unit_gaussian_noise = np.random.normal(0, 1, size = self.d)
        Xy_hat = eX_P.T @ y_P + unit_gaussian_noise * np.sqrt(var_scale)

        # compute the private solution
        beta_ols = np.linalg.inv(XX_hat) @ Xy_hat

        # compute the predicted labels on the public dataset
        y_Q_hat = self.X_Q @ beta_ols 

        for i in range(10):
            cy = y_Q_hat * self.c
            # private hessian
            numer = self.c * phi2(cy).sum() / self.n_Q - 1
            print("numer", numer)
            denorm = phi2(cy).mean() + (cy * phi3(cy)).mean() 
            print("means", phi2(cy).mean(), (cy * phi3(cy)).mean())
            self.c = self.c - numer / denorm
            
            print(numer, denorm, self.c)

        if not math.isnan(self.c):
            self.w_glm = self.c * beta_ols
        else:
            self.w_glm = beta_ols
        return self
    
    def predict_proba(self, X):
        return X @ self.w_glm[1:] + self.w_glm[0]
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


        


        