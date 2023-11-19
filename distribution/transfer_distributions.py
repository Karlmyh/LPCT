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
    return np.abs(np.sin(8 * x[0]) )

def f_1_Q(x):
    return 0.2 * (np.abs(np.sin(8 * x[0])) - 0.5 ) + 0.5 


def f_2_P(x):
    return  0.5 + 0.4  / (1 + np.exp( - 20 * (x[0] - 0.6) ))  * np.cbrt(np.sin( 12.5 * x[1] ))  + 0.1 / (1 + np.exp( - 0.2 * (x[0] - 0.5) ))


def f_2_Q(x):
    return 0.3 * (0.5 + 0.4  / (1 + np.exp( - 20 * (x[0] - 0.6) ))  * np.cbrt(np.sin( 12.5 * x[1] )) + 0.1 / (1 + np.exp( - 1 * (x[0] - 0.5) ))  - 0.5 ) + 0.5 


def f_3_P(x):
    return  (np.ceil(x[0] * 2) +  np.ceil(x[1] * 4)) % 2

def f_3_Q(x):
    return  0.5 * ( (np.ceil(x[0] * 2) +  np.ceil(x[1] * 4)) % 2 - 0.5) + 0.5

def f_4_P(x):
    return  0.5 + 0.4 * np.sign(x[0] - 0.33).astype(float) * np.sign(x[1] - 0.33).astype(float) * ( np.abs(1.5 * x[1] - 0.5) *  np.abs(1.5 * x[0] - 0.5) * np.max([0.0, 1 - np.abs(2 * x[1] - 1),]) )**0.1

def f_4_Q(x):
    return  0.5 + 0.4 * np.sign(x[0] - 0.33).astype(float) * np.sign(x[1] - 0.33).astype(float) * ( np.abs(1.5 * x[1] - 0.5) *  np.abs(1.5 * x[0] - 0.5) * np.max([0.0, 1 - np.abs(2 * x[1] - 1),]) )**0.05


def f_5_P(x):
    return  0.5 + 0.4 * np.sign(x[0] - 0.33).astype(float) * np.sign(x[1] - 0.33).astype(float) * ( np.abs(1.5 * x[1] - 0.5) *  np.abs(1.5 * x[0] - 0.5) * np.max([0.0, 1 - np.abs(2 * x[1] - 1),]) )**0.1

def f_5_Q(x):
    return  0.5 + 0.4 * np.sign(x[0] - 0.33).astype(float) * np.sign(x[1] - 0.33).astype(float) * ( np.abs(1.5 * x[1] - 0.5) *  np.abs(1.5 * x[0] - 0.5) * np.max([0.0, 1 - np.abs(2 * x[1] - 1),]) )**0.5

def f_6_P(x):
    return  0.5 + 0.4 * np.sign(x[0] - 0.33).astype(float) * np.sign(x[1] - 0.33).astype(float) * ( np.abs(1.5 * x[1] - 0.5) *  np.abs(1.5 * x[0] - 0.5) * np.max([0.0, 1 - np.abs(2 * x[1] - 1),]) )**0.1

def f_6_Q(x):
    return  0.5 + 0.4 * np.sign(x[0] - 0.33).astype(float) * np.sign(x[1] - 0.33).astype(float) * ( np.abs(1.5 * x[1] - 0.5) *  np.abs(1.5 * x[0] - 0.5) * np.max([0.0, 1 - np.abs(2 * x[1] - 1),]) )**0.1


def f_7_P(x):
    return  (f_4_P(x[:2]) - 1 / 2) * (f_4_P(x[2:4])  - 1 / 2) * (f_4_P(x[4:6]) - 1 / 2) + 1/2

def f_7_Q(x):
    return  (f_4_Q(x[:2]) - 1 / 2) * (f_4_Q(x[2:4])  - 1 / 2) * (f_4_Q(x[4:6]) - 1 / 2) + 1/2


def f_8_P(x):
    return  0.5 + 0.4 * np.sign(x[0] - 0.33).astype(float) * np.sign(x[1] - 0.33).astype(float) * ( np.abs(1.5 * x[1] - 0.5) *  np.abs(1.5 * x[0] - 0.5) * np.max([0.0, 1 - np.abs(2 * x[1] - 1),]) )**0.1

def f_8_Q(x):
    if x[0] > 0.5:
        return f_4_Q(x)
    else:
        return f_5_Q(x)
    
def f_9_P(x):
    return  0.5 + 0.4 * np.sign(x[0] - 0.33).astype(float) * np.sign(x[1] - 0.33).astype(float) * ( np.abs(1.5 * x[1] - 0.5) *  np.abs(1.5 * x[0] - 0.5) * np.max([0.0, 1 - np.abs(2 * x[1] - 1),]) )**0.1

def f_9_Q(x, gamma = 1):
    return  0.5 + 0.4 * np.sign(x[0] - 0.33).astype(float) * np.sign(x[1] - 0.33).astype(float) * ( np.abs(1.5 * x[1] - 0.5) *  np.abs(1.5 * x[0] - 0.5) * np.max([0.0, 1 - np.abs(2 * x[1] - 1),]) )**(gamma*0.1)



class TransferDistribution(object):
    def __init__(self,index,dim = "auto"):
        self.dim = dim
        self.index = index
        
    def testDistribution_1(self):
        if self.dim == "auto":
            self.dim = 1
        marginal_obj = UniformDistribution(low = np.zeros(self.dim),upper = np.ones(self.dim))
        regression_obj_P = RegressionFunction(f_1_P, self.dim)
        regression_obj_Q = RegressionFunction(f_1_Q, self.dim)
        generation_obj = LabelGeneration()
        
        
        return JointDistribution(marginal_obj, regression_obj_P, regression_obj_Q, generation_obj)
    
    def testDistribution_2(self):
        if self.dim == "auto":
            self.dim = 2
        marginal_obj = UniformDistribution(low = np.zeros(self.dim),upper = np.ones(self.dim))
        regression_obj_P = RegressionFunction(f_2_P, self.dim)
        regression_obj_Q = RegressionFunction(f_2_Q, self.dim)
        generation_obj = LabelGeneration()
        
        
        return JointDistribution(marginal_obj, regression_obj_P, regression_obj_Q, generation_obj)
    
    def testDistribution_3(self):
        if self.dim == "auto":
            self.dim = 2
        marginal_obj = UniformDistribution(low = np.zeros(2),upper = np.ones(2))
        regression_obj_P = RegressionFunction(f_3_P, self.dim)
        regression_obj_Q = RegressionFunction(f_3_Q, self.dim)
        generation_obj = LabelGeneration()
        
        
        return JointDistribution(marginal_obj, regression_obj_P, regression_obj_Q, generation_obj)
    
    
    def testDistribution_4(self):
        if self.dim == "auto":
            self.dim = 2
        marginal_obj = UniformDistribution(low = np.zeros(2),upper = np.ones(2))
        regression_obj_P = RegressionFunction(f_4_P, self.dim)
        regression_obj_Q = RegressionFunction(f_4_Q, self.dim)
        generation_obj = LabelGeneration()
        
        
        return JointDistribution(marginal_obj, regression_obj_P, regression_obj_Q, generation_obj)
    
    
    def testDistribution_5(self):
        if self.dim == "auto":
            self.dim = 2
        marginal_obj = UniformDistribution(low = np.zeros(2),upper = np.ones(2))
        regression_obj_P = RegressionFunction(f_5_P, self.dim)
        regression_obj_Q = RegressionFunction(f_5_Q, self.dim)
        generation_obj = LabelGeneration()
        
        
        return JointDistribution(marginal_obj, regression_obj_P, regression_obj_Q, generation_obj)
    
    def testDistribution_6(self):
        if self.dim == "auto":
            self.dim = 2
        marginal_obj = UniformDistribution(low = np.zeros(2),upper = np.ones(2))
        regression_obj_P = RegressionFunction(f_6_P, self.dim)
        regression_obj_Q = RegressionFunction(f_6_Q, self.dim)
        generation_obj = LabelGeneration()
        
        
        return JointDistribution(marginal_obj, regression_obj_P, regression_obj_Q, generation_obj)
    
    def testDistribution_7(self):
        if self.dim == "auto":
            self.dim = 6
        marginal_obj = UniformDistribution(low = np.zeros(6),upper = np.ones(6))
        regression_obj_P = RegressionFunction(f_7_P, self.dim)
        regression_obj_Q = RegressionFunction(f_7_Q, self.dim)
        generation_obj = LabelGeneration()
        
        
        return JointDistribution(marginal_obj, regression_obj_P, regression_obj_Q, generation_obj)
    

    def testDistribution_8(self):
        if self.dim == "auto":
            self.dim = 2
        marginal_obj = UniformDistribution(low = np.zeros(2),upper = np.ones(2))
        regression_obj_P = RegressionFunction(f_8_P, self.dim)
        regression_obj_Q = RegressionFunction(f_8_Q, self.dim)
        generation_obj = LabelGeneration()
        
        return JointDistribution(marginal_obj, regression_obj_P, regression_obj_Q, generation_obj)
    
    
    def testDistribution_9(self, gamma):
        if self.dim == "auto":
            self.dim = 2
        marginal_obj = UniformDistribution(low = np.zeros(2),upper = np.ones(2))
        regression_obj_P = RegressionFunction(f_9_P, self.dim)
        f_9_modified = lambda x: f_9_Q(x, gamma)
        regression_obj_Q = RegressionFunction(f_9_modified, self.dim)
        generation_obj = LabelGeneration()
        
        return JointDistribution(marginal_obj, regression_obj_P, regression_obj_Q, generation_obj)
    
    def returnDistribution(self, gamma = None):
        switch = {'1': self.testDistribution_1, 
                  '2': self.testDistribution_2,
                  '3': self.testDistribution_3,
                  '4': self.testDistribution_4,
                  '5': self.testDistribution_5,
                  '6': self.testDistribution_6,
                  '7': self.testDistribution_7,
                  '8': self.testDistribution_8,
                  '9': self.testDistribution_9,
          }

        choice = str(self.index)  
        
        if gamma is None:
            result=switch.get(choice)()
        else: 
            result=switch.get(choice)(gamma = gamma)
        return result
    
