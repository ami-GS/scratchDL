import numpy as np
import math
import copy

class Layer(object):
    def __init__(self, input_shape = None, units = None):
        self.input_shape = input_shape
        self.units = units

    def configure(self, shape, phase, prevLayer):
        self.phase = phase
        self.prevLayer = prevLayer

    def forward(self, x):
        pass

    def backward(self, err_delta):
        pass


class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1,1), input_shape = None, padding = 0, dtype=np.float32):
        super(Conv2D, self).__init__(input_shape, None)
        self.filterNum = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.filters = np.random.uniform(-1, 1, (filters, kernel_size**2)).astype(dtype)
        self.dtype = dtype
        self.strides = strides
        if input_shape:
            self.input_shape = (int(math.sqrt(input_shape))+self.padding)**2
            self.units = int(np.sqrt(self.input_shape)) - self.kernel_size + 1
            self.units *= self.units
            self.x_rowcol = int(np.sqrt(self.input_shape))
            self.y_rowcol = self.x_rowcol - self.kernel_size+1

    def configure(self, data_shape, phase, prevLayer=None):
        self.phase = phase
        self.prevLayer = prevLayer
        self.channel = data_shape[1]
        self.batch = data_shape[0]
        if prevLayer:
            self.units = int(np.sqrt(self.input_shape)) - self.kernel_size + 1
            self.units *= self.units
            self.input_shape = (int(math.sqrt(prevLayer.units)) + self.padding)**2
        self.x_rowcol = int(np.sqrt(self.input_shape))
        self.y_rowcol = (self.x_rowcol + 2 * self.padding - self.kernel_size)/self.strides[0] + 1
        self.X = np.zeros((self.batch, self.channel, self.kernel_size**2, self.y_rowcol**2), dtype=self.dtype)
        self.Y = np.zeros((self.batch, self.filterNum, self.y_rowcol**2), dtype=self.dtype)
        if phase == "TRAIN":
            self.E = np.zeros((self.batch, self.filterNum, self.y_rowcol**2), dtype=self.dtype)
            self.err_delta = np.zeros((self.batch, self.channel, self.x_rowcol, self.x_rowcol), dtype=self.dtype)

    def forward(self, x):
        self.Y = np.zeros((self.batch, self.filterNum, self.y_rowcol**2), dtype=self.dtype)
        x = x.reshape(self.batch, self.channel, self.x_rowcol-self.padding, self.x_rowcol-self.padding)
        if self.padding:
            x = np.lib.pad(x, (1,1), 'constant', constant_values=(0,0))

        for i in range(0, self.y_rowcol, self.strides[0]):
            for j in range(0, self.y_rowcol, self.strides[1]):
                self.X[:, :, :, self.y_rowcol*i+j%self.y_rowcol] = \
                x[:, :, i:i+self.kernel_size, j:j+self.kernel_size].reshape(self.batch, self.channel, self.kernel_size**2)

        for b in range(self.batch):
            for c in range(self.channel):
                self.Y[b, :] += self.filters.dot(self.X[b,c])
        return self.Y.reshape(self.batch, self.filterNum, self.y_rowcol, self.y_rowcol)

    def backward(self, err_delta):
        self.E[:] = err_delta.reshape(self.batch, self.filterNum, self.y_rowcol**2)

        for yi in range(self.y_rowcol):
            for yj in range(self.y_rowcol):
                # not good for vector processing
                tmp = np.einsum("bf,fk->bfk", self.E[:,:,yi*self.y_rowcol + yj%self.y_rowcol], self.filters)
                for f in range(self.filterNum):
                    yyi = yi * self.strides[0]
                    yyj = yj * self.strides[1]
                    self.err_delta[:, :, yyi:yyi+self.kernel_size, yyj:yyj+self.kernel_size] += \
                        tmp[:,f,:].reshape(self.batch, 1, self.kernel_size, self.kernel_size)

        tmp = np.zeros(self.filters.shape)
        for b in range(self.batch):
            for c in range(self.channel):
                tmp += self.X[b,c,:,:].dot(self.E[b,:,:].T).T
        self.filters -= tmp/self.batch

        return self.err_delta

class MaxPooling2D(Layer):
    def __init__(self, kernel_size, strides=(1,1), dtype=np.float32):
        super(MaxPooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.dtype = dtype

    def configure(self, data_shape, phase, prevLayer = None):
        self.phase = phase
        self.prevLayer = prevLayer
        # TODO : not good,
        self.x_rowcol = prevLayer.prevLayer.y_rowcol
        self.input_shape = self.x_rowcol**2
        self.y_rowcol = self.x_rowcol - self.kernel_size+1
        self.channel = prevLayer.prevLayer.filterNum
        self.batch = data_shape[0]
        self.X = np.zeros((self.batch, self.channel, self.kernel_size**2, self.y_rowcol**2), dtype=self.dtype)
        if phase == "TRAIN":
            self.maxLocs = np.zeros((self.batch, self.channel, self.y_rowcol**2), dtype=int)
            self.E = np.zeros((self.batch, self.channel, self.y_rowcol**2), dtype=self.dtype)
            self.err_delta = np.zeros((self.batch, self.channel, self.x_rowcol, self.x_rowcol), dtype=self.dtype)
            self.Y = np.zeros((self.batch, self.channel, self.y_rowcol**2), dtype=self.dtype)

    def forward(self, x):
        for i in range(0, self.y_rowcol, self.strides[0]):
            for j in range(0, self.y_rowcol, self.strides[1]):
                self.X[:, :, :, self.y_rowcol*i+j%self.y_rowcol] = \
                x[:, :, i:i+self.kernel_size, j:j+self.kernel_size].reshape(self.batch, self.channel, self.kernel_size**2)

        if self.phase == "TRAIN":
            self.maxLocs[:] = self.X.argmax(axis=2)
        return self.X.max(axis=2).reshape(self.batch, self.channel, self.y_rowcol, self.y_rowcol)
        #return self.Y.reshape(self.batch, self.channel, self.y_rowcol, self.y_rowcol)

    def backward(self, err_delta):
        self.E[:] = err_delta.reshape(self.batch, self.channel, self.y_rowcol**2)

        for b in range(self.batch):
            for c in range(self.channel):
                for y in range(self.y_rowcol**2):
                    # TODO : really bad way
                    loc = self.maxLocs[b,c,y]
                    locy = loc / self.kernel_size + (y / self.y_rowcol)
                    locx = loc % self.kernel_size + (y % self.y_rowcol)
                    self.err_delta[b,c,locx,locy] += self.E[b,c,y]

        return self.err_delta


class FullyConnect(Layer):
    def __init__(self, units, input_shape=0, dtype=np.float32):
        super(FullyConnect, self).__init__(input_shape, units)
        self.W = np.random.uniform(-1, 1, (input_shape, units)).astype(dtype)
        # TODO : sharing bias to all batch
        self.bias = np.random.uniform(-1, 1, 1).astype(dtype)
        self.original_shape = None
        self.dtype = dtype

    def configure(self, data_shape, phase, prevLayer = None):
        self.phase = phase
        self.prevLayer = prevLayer
        self.batch = data_shape[0]
        if isinstance(prevLayer, MaxPooling2D):
            self.input_shape = prevLayer.channel * prevLayer.y_rowcol * prevLayer.y_rowcol
        if phase == "TRAIN":
            self.X = np.zeros((self.batch, self.input_shape), dtype=self.dtype)
            self.E = np.zeros((self.batch, self.units), dtype=self.dtype)
            self.Y = np.zeros((self.batch, self.units), dtype=self.dtype)

    def forward(self, x):
        # for 2D data (like image)
        if len(x.shape) > 1:
            self.original_shape = x.shape
            x = x.reshape(self.original_shape[0], reduce(lambda x,y:x*y, self.original_shape[1:]))

        if self.phase == "TRAIN":
            self.X[:] = x
        return x.dot(self.W) + self.bias

    def backward(self, err_delta):
        self.E[:] = err_delta
        err_delta = self.E.dot(self.W.T)

        # updates
        np.subtract(self.W, self.optimizer(np.sum(np.einsum("bi,bj->bij", self.X, self.learning_rate*self.E), axis=0))/self.batch, self.W)
        self.bias -= np.sum(self.learning_rate * self.E)/self.batch

        if self.original_shape:
            err_delta = err_delta.reshape(self.original_shape)
        return err_delta

from activation import Sigmoid, Tanh
class LSTM(Layer):
    def __init__(self, units=0, input_shape=0, dtype=np.float32, gate_act=Sigmoid):
        super(LSTM, self).__init__(input_shape, units)
        self.gate_act = {"I":gate_act(), "F":gate_act(), "O":gate_act()}
        self.tanh = {"U":Tanh(), "C":Tanh()}
        self.Wx = {"I":None, "F":None, "U":None, "O":None}
        self.Wh = {"I":None, "F":None, "U":None, "O":None}
        self.B = {"I":None, "F":None, "U":None, "O":None}
        for k in ["I", "F", "U", "O"]:
            self.Wx[k] = np.random.uniform(-1, 1, (input_shape, units)).astype(dtype)
            self.Wh[k] = np.random.uniform(-1, 1, (units, units)).astype(dtype)
            self.B[k] = np.random.uniform(-1, 1, 1).astype(dtype)

    def configure(self, data_shape, phase, prevLayer = None):
        self.batch = data_shape[0]
        for k in self.gate_act:
            self.gate_act[k].configure(data_shape, phase, prevLayer)
        for k in self.tanh:
            self.tanh[k].configure(data_shape, phase, prevLayer)
        self.optimizers = []
        for i in range(8):
            self.optimizers.append(copy.deepcopy(self.optimizer))
        self.buff = {"C":None, "C_1":None, "H":None, "H_1":None, "I":None, "F":None, "U":None, "O":None, "X":None}
        for k in self.buff:
            self.buff[k] = np.zeros((self.batch, self.units))
        self.X = np.zeros((self.batch, self.input_shape), dtype=self.dtype)

    def forward(self, x):
        self.X[:] = x

        for k in ["I", "F", "O"]:
            self.buff[k] = self.gate_act[k].forward(self.X.dot(self.Wx[k]) + self.buff["H_1"].dot(self.Wh[k]) + self.B[k])
        self.buff["U"] = self.tanh["U"].forward(self.X.dot(self.Wx["U"]) + self.buff["H_1"].dot(self.Wh["U"]) + self.B["U"])
        self.buff["C"] = self.buff["I"]*self.buff["C_1"] + self.buff["U"] * self.buff["I"]
        self.Ctanh = self.tanh["C"].forward(self.buff["C"])
        self.buff["H"] = self.Ctanh * self.buff["O"]
        self.buff["C_1"] = self.buff["C"]
        self.buff["H_1"] = self.buff["H"]
        return self.buff["H"]

    def backward(self, e):
        delta = {}
        delta["C"] = self.tanh["C"].backward(e)*self.buff["O"]
        delta["C_1"] = delta["C"] * self.buff["F"]
        delta["O"] = self.gate_act["O"].backward(e)*self.Ctanh
        delta["I"] = self.gate_act["I"].backward(delta["C"])*self.buff["U"]
        delta["U"] = self.tanh["U"].backward(delta["C"])*self.buff["I"]
        delta["F"] = self.gate_act["F"].backward(delta["C"])*self.buff["C_1"]
        delta["H"] = delta["I"].dot(self.Wh["I"].T) + delta["O"].dot(self.Wh["O"].T) + delta["U"].dot(self.Wh["U"].T) + delta["F"].dot(self.Wh["F"].T)

        #update
        for i, k in enumerate(self.Wx):
            np.subtract(self.Wx[k], self.optimizers[i](np.sum(np.einsum("bi,bj->bij", self.X, self.learning_rate*delta[k]), axis=0))/self.batch, self.Wx[k])

        for i, k in enumerate(self.Wh):
            np.subtract(self.Wh["I"], self.optimizers[4+i](np.sum(np.einsum("bi,bj->bij", self.buff["H_1"], self.learning_rate*delta["I"]), axis=0))/self.batch, self.Wh["I"])

        for k in self.B:
            self.B[k] -= np.sum(self.learning_rate * delta[k])

        return delta["H"]


class BatchNorm(Layer):
    def __init__(self, units=0, input_shape=0, dtype=np.float32):
        super(BatchNorm, self).__init__(input_shape, units)
        self.beta = None
        self.gamma = None
        self.sub = None
        self.sq = None
        self.invsq = None
        self.var = None
        self.xhat = None
        self.epsilon = 10e-12

    def configure(self, data_shape, phase, prevLayer = None):
        self.phase = phase
        self.prevLayer = prevLayer
        self.input_shape = prevLayer.input_shape
        self.units = prevLayer.units
        self.batch = data_shape[0]
        self.beta = 0
        self.gamma = 1

    def forward(self, x):
        mean = np.sum(x, axis=0)/self.batch
        self.sub = x - mean
        self.var = np.var(x, axis=0)
        self.sq = np.sqrt(self.var + self.epsilon)
        self.invsq = 1/self.sq
        self.xhat = self.sub*self.invsq
        return self.gamma * self.xhat + self.beta

    def backward(self, e):
        dxhat = e * self.gamma
        dvar = np.sum(dxhat * self.sub * self.sq**(1.5)/-2, axis=0)
        dmean = np.sum(dxhat * - self.invsq, axis=0) + dvar * -2 * self.var
        dx = dxhat * self.invsq + dvar * -2 * self.var + dmean / self.batch
        db = np.sum(e, axis=0)
        dg = db * self.xhat
        self.beta -= db
        self.gamma -= dg
        return dx