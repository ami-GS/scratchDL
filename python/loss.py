from layer import Layer
import numpy as np

class Loss(object):
    def __init__(self):
        pass

    def calc(self):
        pass

    def partial_derivative(self):
        pass

class MSE(Loss):
    def __init__(self):
        super(MSE, self).__init__()

    def calc(self, X, label, batch=1):
        if batch==1:
            if len(X.shape) != 1 or (len(label.shape) != 1 and label.shape[0] != 1):
                print "loss error"

        return np.sum(np.power(np.abs(X-label), 2))*0.5

    def partial_derivative(self, X, label):
        return - (label - X)