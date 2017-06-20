import numpy as np
import math

class Layer(object):
    def __init__(self, input_shape = None, units = None):
        self.input_shape = input_shape
        self.units = units
        # input
        self.X = np.zeros(input_shape)
        # calculated value
        self.Y = np.zeros(units)
        # error
        self.E = np.zeros(units)

    def configure(self, shape, prevLayer):
        self.prevLayer = prevLayer

    def forward(self, x):
        pass

    def backward(self, err_delta):
        pass


class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1,1), input_shape = None, padding = 0):
        super(Conv2D, self).__init__(input_shape, None)
        self.filterNum = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.filters = np.random.uniform(-1, 1, (filters, kernel_size**2))
        self.strides = strides
        if input_shape:
            self.input_shape = (int(math.sqrt(input_shape))+self.padding)**2
            self.units = int(np.sqrt(self.input_shape)) - self.kernel_size + 1
            self.units *= self.units
            self.x_rowcol = int(np.sqrt(self.input_shape))
            self.y_rowcol = self.x_rowcol - self.kernel_size+1

    def configure(self, data_shape, prevLayer=None):
        self.prevLayer = prevLayer
        self.channel = data_shape[1]
        self.batch = data_shape[0]
        if prevLayer:
            self.units = int(np.sqrt(self.input_shape)) - self.kernel_size + 1
            self.units *= self.units
            self.input_shape = (int(math.sqrt(prevLayer.units)) + self.padding)**2
        self.x_rowcol = int(np.sqrt(self.input_shape))
        self.y_rowcol = (self.x_rowcol + 2 * self.padding - self.kernel_size)/self.strides[0] + 1
        self.X = np.zeros((self.batch, self.channel, self.kernel_size**2, self.y_rowcol**2))
        self.E = np.zeros((self.batch, self.filterNum, self.y_rowcol**2))
        self.err_delta = np.zeros((self.batch, self.channel, self.x_rowcol, self.x_rowcol))
        self.Y = np.zeros((self.batch, self.filterNum, self.y_rowcol**2))

    def forward(self, x):
        x = x.reshape(self.batch, self.channel, self.x_rowcol-self.padding, self.x_rowcol-self.padding)
        if self.padding:
            x = np.lib.pad(x, (1,1), 'constant', constant_values=(0,0))
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
                self.err_delta[:, :, yi:yi+self.kernel_size, yj:yj+self.kernel_size] += \
                        np.sum(np.einsum("bf,fk->bfk", self.E[:,:,yi*self.y_rowcol + yj%self.y_rowcol], self.filters), axis=1)\
                          .reshape(self.batch, self.channel, self.kernel_size, self.kernel_size)

        tmp = np.zeros(self.filters.shape)
        for b in range(self.batch):
            for c in range(self.channel):
                tmp += self.X[b,c,:,:].dot(self.E[b,:,:].T).T
        self.filters -= tmp/self.batch

        return self.err_delta

class MaxPooling2D(Layer):
    def __init__(self, kernel_size, strides=(1,1)):
        super(MaxPooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides

    def configure(self, data_shape, prevLayer = None):
        self.prevLayer = prevLayer
        # TODO : not good,
        self.x_rowcol = prevLayer.prevLayer.y_rowcol
        self.input_shape = self.x_rowcol**2
        self.y_rowcol = self.x_rowcol - self.kernel_size+1
        self.channel = prevLayer.prevLayer.filterNum
        self.batch = data_shape[0]
        self.maxLocs = np.zeros((self.batch, self.channel, self.y_rowcol, self.y_rowcol, 2), dtype=int)
        self.X = np.zeros((self.batch, self.channel, self.x_rowcol, self.x_rowcol))
        self.E = np.zeros((self.batch, self.channel, self.y_rowcol, self.y_rowcol))
        self.err_delta = np.zeros((self.batch, self.channel, self.x_rowcol, self.x_rowcol))
        self.Y = np.zeros((self.batch, self.channel, self.y_rowcol, self.y_rowcol))

    def forward(self, x):
        self.X[:] = x
        for c in range(self.channel):
            for yi in range(0, self.y_rowcol, self.strides[0]):
                for yj in range(0, self.y_rowcol, self.strides[1]):
                    tmp = self.X[:, c, yi:yi+self.kernel_size, yj:yj+self.kernel_size].reshape(self.batch, -1).argmax(1)
                    self.maxLocs[:, c, yi, yj, 0] = (tmp/self.kernel_size).astype(int)
                    self.maxLocs[:, c, yi, yj, 1] = tmp%self.kernel_size
                    self.Y[:, c, yi, yj] = np.max(self.X[:, c, yi:yi+self.kernel_size, yj:yj+self.kernel_size], axis=(1,2))
        return self.Y

    def backward(self, err_delta):
        self.E[:] = err_delta

        for b in range(self.batch):
            for c in range(self.channel):
                for yi in range(self.y_rowcol):
                    for yj in range(self.y_rowcol):
                        # TODO : really bad way
                        self.err_delta[b,c,yi+self.maxLocs[b,c,yi,yj,0], yj+self.maxLocs[b,c,yi,yj,1]] += self.E[b,c,yi,yj]

        return self.err_delta


class FullyConnect(Layer):
    def __init__(self, units, input_shape=0):
        super(FullyConnect, self).__init__(input_shape, units)
        self.W = np.random.uniform(-1, 1, (input_shape, units))
        # TODO : sharing bias to all batch
        self.bias = np.random.uniform(-1, 1, 1)
        self.original_shape = None

    def configure(self, data_shape, prevLayer = None):
        self.prevLayer = prevLayer
        self.batch = data_shape[0]
        if isinstance(prevLayer, MaxPooling2D):
            self.input_shape = prevLayer.channel * prevLayer.y_rowcol * prevLayer.y_rowcol
        self.X = np.zeros((self.batch, self.input_shape))
        self.E = np.zeros((self.batch, self.units))
        self.Y = np.zeros((self.batch, self.units))

    def forward(self, x):
        # for 2D data (like image)
        if len(x.shape) > 1:
            self.original_shape = x.shape
            x = x.reshape(self.original_shape[0], reduce(lambda x,y:x*y, self.original_shape[1:]))

        self.X[:] = x
        self.Y = x.dot(self.W) + self.bias
        return self.Y

    def backward(self, err_delta):
        self.E[:] = err_delta
        err_delta = self.E.dot(self.W.T)

        # updates
        np.subtract(self.W, self.optimizer(np.sum(np.einsum("bi,bj->bij", self.X, self.learning_rate*self.E), axis=0))/self.batch, self.W)
        self.bias -= np.sum(self.learning_rate * self.E)

        if self.original_shape:
            err_delta = err_delta.reshape(self.original_shape)
        return err_delta