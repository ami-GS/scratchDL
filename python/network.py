import numpy as np
from layer import MaxPooling2D, Conv2D
from activation import Activation
from optimizer import PassThrough
from loss import MSE
import copy

class Network:
    def __init__(self, layers=None, batch=1, learning_rate=0.02, optimizer=PassThrough()):
        self.layers = layers
        self.batch = batch
        self.learning_rate = learning_rate
        self.input_shape = layers[0].input_shape
        units = self.layers[0].units
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.learning_rate = learning_rate
            layer.batch = batch
            layer.optimizer = copy.deepcopy(optimizer)
            if i >= 1:
                layer.input_shape = units
                if isinstance(layer, Activation):
                    layer.units = units
                elif isinstance(layer, MaxPooling2D) or isinstance(layer, Conv2D):
                    tmp = layer.input_shape - layer.kernel_size + 1
                    layer.Y = np.zeros((1, tmp, tmp))
                    layer.units = tmp**2
            if self.batch == 1:
                layer.X = np.zeros(layer.input_shape)
                layer.Y = np.zeros(layer.units)
                layer.E = np.zeros(layer.units)
            else:
                layer.X = np.zeros((batch, layer.input_shape))
                layer.Y = np.zeros((batch, layer.units))
                layer.E = np.zeros((batch, layer.units))
            units = layer.units
        # current value
        self.Y = None
        self.last_units = units

    def predict(self, X):
        if len(X.shape) > 1 and self.batch:
            ans = np.zeros((X.shape[0], self.last_units))
            for i in range(0, X.shape[0], self.batch):
                tmp = X[i:i+self.batch, :]
                for layer in self.layers:
                    tmp = layer.forward(tmp)
                ans[i:i+self.batch, :] = tmp
            return ans
        else:
            self.Y = X
            for layer in self.layers:
                self.Y = layer.forward(self.Y)
        return self.Y

    def train(self, X, label, loss=MSE()):
        self.Y = X
        for layer in self.layers:
            self.Y = layer.forward(self.Y)

        err = loss.calc(self.Y, label, self.batch)
        err_delta = loss.partial_derivative(self.Y, label)
        for layer in self.layers[::-1]:
            err_delta = layer.backward(err_delta)

        return err
