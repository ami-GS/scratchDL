import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self):
        super(Activation, self).__init__()

class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        # destructive assignment
        self.Y = np.reciprocal(1 + np.exp(-x))

        return self.Y

    def backward(self, err_delta):
        self.E = err_delta
        err_delta = np.multiply(err_delta, np.multiply(self.Y, 1-self.Y))
        return err_delta

class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        # destructive assignment
        self.Y = np.multiply(np.greater_equal(x, 0), x)
        return self.Y

    def backward(self, err_delta):
        self.E = err_delta
        err_delta *= np.greater(self.Y, 0)*1
        return err_delta

class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        # destructive assignment
        self.Y = np.tanh(x)

        return self.Y

    def backward(self, err_delta):
        self.E = err_delta
        np.multiply(err_delta, 1 - np.power(self.Y, 2), err_delta)

        return err_delta


class Softmax(Activation):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        original_shape = x.shape
        if len(original_shape) >= 3:
            # TODO : hardcoding
            x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
        exp = np.exp(x)
        if self.batch > 1:
            r_expsum = np.reciprocal(np.sum(exp, axis=1))
        else:
            r_expsum = np.reciprocal(np.sum(exp)+0.000000001)
        self.Y = (exp.T*r_expsum).T

        if len(original_shape) >= 3:
            self.Y = self.Y.reshape(original_shape)

        return self.Y

    def backward(self, err_delta):
        # pass through?
        self.E = err_delta
        return err_delta