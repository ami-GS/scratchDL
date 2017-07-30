import numpy as np
from layer import MaxPooling2D, Conv2D
from activation import Activation
from optimizer import PassThrough
from loss import MSE
import copy

class Network:
    def __init__(self, layers=None, batch=1, learning_rate=0.02, optimizer=PassThrough(), dtype=np.float32):
        self.layers = layers
        self.batch = batch
        self.learning_rate = learning_rate
        self.input_shape = layers[0].input_shape
        self.configured = False
        self.dtype = dtype

        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.learning_rate = learning_rate
            layer.batch = batch
            layer.dtype = dtype
            layer.optimizer = copy.deepcopy(optimizer)
        self.last_units = layers[-1].units
        if isinstance(layers[-1], Activation):
            self.last_units = layers[-2].units

    def predict(self, X, batch=1):
        prevLayer = None
        for l in self.layers:
            l.configure(X.shape, "TEST", prevLayer)
            prevLayer = l

        ans = np.zeros((X.shape[0], self.last_units))
        for i in range(0, X.shape[0], self.batch):
            tmp = X[i:i+self.batch, :]
            for layer in self.layers:
                tmpbatch = layer.batch
                layer.batch = batch
                tmp = layer.forward(tmp)
                layer.batch = tmpbatch
            ans[i:i+self.batch, :] = tmp
        return ans

    def train(self, X, label, loss=MSE()):
        if self.configured == False:
            prevLayer = None
            for l in self.layers:
                l.configure(X.shape, "TRAIN", prevLayer)
                prevLayer = l
            self.configured = True

        self.Y = X
        for layer in self.layers:
            self.Y = layer.forward(self.Y)

        err = loss.calc(self.Y, label, self.batch)
        err_delta = loss.partial_derivative(self.Y, label)
        for layer in self.layers[::-1]:
            err_delta = layer.backward(err_delta)

        return err
