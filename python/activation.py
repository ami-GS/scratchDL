import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self):
        super(Activation, self).__init__()

class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        if self.phase == "TRAIN":
            # destructive assignment
            self.Y = np.reciprocal(1 + np.exp(-x))
            return self.Y
        else:
            return np.reciprocal(1 + np.exp(-x))

    def backward(self, err_delta):
        return np.multiply(err_delta, np.multiply(self.Y, 1-self.Y))

class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        if self.phase == "TRAIN":
            # destructive assignment
            self.Y = np.multiply(np.greater_equal(x, 0), x)
            return self.Y
        else:
            return np.multiply(np.greater_equal(x, 0), x)

    def backward(self, err_delta):
        return err_delta * np.greater(self.Y, 0)*1

class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        if self.phase == "TRAIN":
            # destructive assignment
            self.Y = np.tanh(x)
            return self.Y
        else:
            return np.tanh(x)

    def backward(self, err_delta):
        return np.multiply(err_delta, 1 - np.power(self.Y, 2))


class Softmax(Activation):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        exp = np.exp(x.T - np.max(x, axis=1)).T
        r_expsum = np.reciprocal(np.sum(exp, axis=1)+10e-20)

        return (exp.T*r_expsum).T

    def backward(self, err_delta):
        # pass through?
        self.E = err_delta
        return err_delta