import numpy as np
from scipy import stats as st
from scipy import special as sc
import math
import random
import torch


def PrivUnitG(x, eps, C, p):
    """
    Inject noise with privacy budget eps and clipping parameter C by PrivUnitG.
    See https://github.com/optimization-for-data-driven-science/DP-with-public-data
    for the original code. 
    See https://arxiv.org/pdf/2306.15056.pdf for details.
    """
    gamma, sigma = get_gamma_sigma(p, eps)
    param_shape = x.shape
   
    x = x.ravel()
    x = x / max(torch.norm(x), C)
    dim = x.size(- 1)
    g = np.random.normal(0, 1, size = dim)
    g = torch.from_numpy(g).double()
    pos_cor = np.random.binomial(1, p)
    
    
    if pos_cor:
        chosen_dps = np.array([sample_from_G_tail_stable(gamma)])
    else:
        dps = np.random.normal(0, 1, size = 1000)
        chosen_dps = dps[dps<gamma]
    
    if chosen_dps.size == 0:
        print('failure')
        return g * sigma
    target_dp = chosen_dps[0]

    # target_dp seems to be alpha

    
    yperp = g - (x.ravel().dot(g)) * x
    ypar = target_dp * x

    
    return sigma * (yperp + ypar).reshape(param_shape)




def get_gamma_sigma(p, eps):
    # Want p(1-q)/q(1-p) = exp(eps)
    # I.e q^{-1} -1 = (1-q)/q = exp(eps) * (1-p)/p
    if eps < 20:
        qinv = 1 + (math.exp(eps) * (1.0-p)/ p)
    else:
        qinv = 4e8
    q = 1.0 / qinv
    gamma = st.norm.isf(q)
    # Now the expected dot product is (1-p)*E[N(0,1)|<gamma] + pE[N(0,1)|>gamma]
    # These conditional expectations are given by pdf(gamma)/cdf(gamma) and pdf(gamma)/sf(gamma)
    unnorm_mu = st.norm.pdf(gamma) * (-(1.0-p)/st.norm.cdf(gamma) + p/st.norm.sf(gamma))
    sigma = 1./unnorm_mu
    return gamma, sigma


def priv_unit_G_get_p(eps):
    # Mechanism:
    # With probability p, sample a Gaussian conditioned on g.x \geq gamma
    # With probability (1-p), sample conditioned on g.x \leq gamma
    # Scale g appropriately to get the expectation right
    # Let q(gamma) = Pr[g.x \geq gamma] = Pr[N(0,1) \geq gamma] = st.norm.sf(gamma)
    # Then density for x above threshold = p(x)  * p/q(gamma)
    # And density for x below threhsold = p(x) * (1-p)/(1-q(gamma))
    # Thus for a p, gamma is determined by the privacy constraint.
    plist = np.arange(0.01, 1.0, 0.01)
    glist = []
    slist = []
    for p in plist:
        gamma, sigma = get_gamma_sigma(p, eps)
        # thus we have to scale this rv by sigma to get it to be unbiased
        # The variance proxy is then d sigma^2
        slist.append(sigma)
        glist.append(gamma)
    ii = np.argmin(slist)

    return plist[ii]



# More stable version. Works at least until 1000
def sample_from_G_tail_stable(gamma):
    # return sample_from_G_tail(gamma)
    logq = st.norm.logsf(gamma)
    u = np.random.uniform(low=0, high=1)
    logu = np.log(u)
    logr = logq + logu # r is now uniform in (0,q)
    #print(q,r)
    return -sc.ndtri_exp(logr)