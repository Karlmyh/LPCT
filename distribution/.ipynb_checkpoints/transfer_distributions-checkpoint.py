from .marginal_distributions import (LaplaceDistribution, 
                          BetaDistribution,
                          DeltaDistribution,
                          MultivariateNormalDistribution,
                          UniformDistribution,
                          MarginalDistribution,
                          ExponentialDistribution,
                          MixedDistribution,
                          UniformCircleDistribution,
                          CauchyDistribution,
                          CosineDistribution,
                          TDistribution
                          )

from .regression_function import RegressionFunction

from .label_generation_distribution import LabelGeneration

from .joint_distribution import JointDistribution

import numpy as np
import math



def f_1_P(x):
    return np.abs(np.sin(16 * x[0]) )

def f_1_Q(x):
    return 0.5 * (np.abs(np.sin(16 * x[0])) - 0.5 ) + 0.5 


def f_2_P(x):
    return np.abs(np.sin(16 * x[0]) * np.sin(16 * x[1]) )


def f_2_Q(x):
    return 0.1 * (np.abs(np.sin(16 * x[0]) * np.sin(16 * x[1]) ) - 0.5 ) + 0.5 





class TransferDistribution(object):
    def __init__(self,index,dim = "auto"):
        self.dim = dim
        self.index = index
        
    def testDistribution_1(self):
        if self.dim == "auto":
            self.dim = 2
        marginal_obj = MultivariateNormalDistribution(0.5,0.01)
        regression_obj_P = RegressionFunction(f_1_P, self.dim)
        regression_obj_Q = RegressionFunction(f_1_Q, self.dim)
        generation_obj = LabelGeneration()
        
        
        return JointDistribution(marginal_obj, regression_obj_P, regression_obj_Q, generation_obj)
    
    def testDistribution_2(self):
        if self.dim == "auto":
            self.dim = 2
        marginal_obj = UniformDistribution(low = np.ones(2),upper = np.ones(2))
        regression_obj_P = RegressionFunction(f_2_P, self.dim)
        regression_obj_Q = RegressionFunction(f_2_Q, self.dim)
        generation_obj = LabelGeneration()
        
        
        return JointDistribution(marginal_obj, regression_obj_P, regression_obj_Q, generation_obj)

    def returnDistribution(self):
        switch = {'1': self.testDistribution_1, 
                  '2': self.testDistribution_2, 
          }

        choice = str(self.index)  
                     
        result=switch.get(choice)()
        return result
    
