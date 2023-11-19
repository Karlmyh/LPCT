from distribution import TransferDistribution
import numpy as np

def test_transferdistribution():
    distribution = TransferDistribution(1).returnDistribution()
    
    
    X_P, y_P_true, X_Q, y_Q_true = distribution.generat_true(1000, 2000)
    
    y_P_true_evaluated, _ = distribution.evaluate(X_P = X_P)
    assert np.array_equal(y_P_true_evaluated, y_P_true)
    
    y_P_true_evaluated, y_Q_true_evaluated = distribution.evaluate(X_P = X_P, X_Q = X_P)
    
    y_bayes_P = (y_P_true_evaluated > 0.5).astype(int)
    y_bayes_Q = (y_Q_true_evaluated > 0.5).astype(int)
    assert np.array_equal(y_bayes_P, y_bayes_Q)
    
    
