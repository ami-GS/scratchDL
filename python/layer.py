import numpy as np

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

    def forward(self, x):
        pass

    def backward(self, err_delta):
        pass


class Conv2D(Layer):
    def __init__(self, filters, channel, kernel_size, strides=(1,1), input_shape = None):
        super(Conv2D, self).__init__(input_shape, None)
        self.filterNum = filters
        self.kernel_size = kernel_size
        self.filters = np.random.uniform(-1, 1, (filters, kernel_size, kernel_size))
        self.channel = 1 # WIP
        self.strides = strides
        # size is valid
        self.units = int(np.sqrt(self.input_shape)) - self.kernel_size + 1
        self.units *= self.units
        if input_shape:
            self.x_rowcol = int(np.sqrt(self.input_shape))
            self.y_rowcol = self.x_rowcol - self.kernel_size+1

    def forward(self, x):
        self.X = x
        if len(x.shape) >= 2:
            self.channel = x.shape[0]

        if self.x_rowcol:
            self.x_rowcol = int(np.sqrt(self.input_shape))
            self.y_rowcol = self.x_rowcol - self.kernel_size+1

        self.X = np.reshape(self.X, (self.channel, self.x_rowcol, self.x_rowcol))
        self.Y = np.zeros((self.filterNum, self.y_rowcol, self.y_rowcol))
        for f in range(self.filterNum):
            for c in range(self.channel):
                for yi in range(0, self.y_rowcol, self.strides[0]):
                    for yj in range(0, self.y_rowcol, self.strides[1]):
                        self.Y[f,yi,yj] = np.sum(np.multiply(self.X[c,yi:yi+self.kernel_size,yj:yj+self.kernel_size], self.filters[f,:,:]))

        return self.Y

    def backward(self, err_delta):
        if len(err_delta.shape) != 3:
            err_delta = err_delta.reshpae((self.filterNum, self.y_rowcol, self.y_rowcol))
        self.E = err_delta
        err_delta = np.zeros((self.channel, self.x_rowcol, self.x_rowcol))

        for yi in range(self.y_rowcol):
            for yj in range(self.y_rowcol):
                err_delta[:, yi:yi+self.kernel_size, yj:yj+self.kernel_size] = np.add(
                    err_delta[:, yi:yi+self.kernel_size, yj:yj+self.kernel_size],
                    np.sum((self.E[:,yi,yj] * self.filters.T).T, axis=0))

        for yi in range(0, self.y_rowcol, self.strides[0]):
            for yj in range(0, self.y_rowcol, self.strides[1]):
                self.filters -= self.optimizer(
                    self.learning_rate *
                    np.sum(np.sum(
                            np.outer(self.E[:,yi,yj],
                                     self.X[:,yi:yi+self.kernel_size, yj:yj+self.kernel_size]),
                            axis=0).reshape((self.channel, self.kernel_size, self.kernel_size)),
                        axis=0))

        return err_delta

class MaxPooling2D(Layer):
    def __init__(self, kernel_size, strides=(1,1)):
        super(MaxPooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides

    def forward(self, x):
        self.X = x
        self.x_rowcol = int(np.sqrt(self.input_shape))
        self.y_rowcol = self.x_rowcol - self.kernel_size+1
        self.channel = x.shape[0]
        self.maxLocs = np.zeros((self.channel, self.y_rowcol, self.y_rowcol, 2))
        # TODO: workaround
        self.Y = np.zeros((self.channel, self.y_rowcol, self.y_rowcol))

        for c in range(self.channel):
            for yi in range(0, self.y_rowcol, self.strides[0]):
                for yj in range(0, self.y_rowcol, self.strides[1]):
                    tmp = np.argmax(self.X[c, yi:yi+self.kernel_size, yj:yj+self.kernel_size])
                    self.maxLocs[c, yi, yj, :] = [tmp%self.kernel_size, tmp/self.kernel_size]
                    self.Y[c, yi, yj] = np.max(self.X[c, yi:yi+self.kernel_size, yj:yj+self.kernel_size])
        return self.Y

    def backward(self, err_delta):
        if len(err_delta.shape) != 3:
            err_delta = err_delta.reshpae((self.channel, self.y_rowcol, self.y_rowcol))
        self.E = err_delta
        err_delta = np.zeros((self.channel, self.x_rowcol, self.x_rowcol))

        for c in range(self.channel):
            for yi in range(self.y_rowcol):
                for yj in range(self.y_rowcol):
                    err_delta[c, yi+self.maxLocs[c,yi,yj,0], yj+self.maxLocs[c,yi,yj,1]] += self.E[c, yi, yj]

        return err_delta


class FullyConnect(Layer):
    def __init__(self, units, input_shape):
        super(FullyConnect, self).__init__(input_shape, units)
        self.W = np.random.uniform(-1, 1, (input_shape, units))
        # TODO : sharing bias to all batch
        self.bias = np.random.uniform(-1, 1, 1)
        self.original_shape = None

    def forward(self, x):
        # for 2D data (like image)
        # batch == 0 is workaround
        if len(x.shape) > 1 and self.batch == 1:
            self.original_shape = x.shape
            x = np.ravel(x)
        self.X = x
        self.Y = x.dot(self.W) + self.bias
        return self.Y

    def backward(self, err_delta):
        self.E = err_delta
        err_delta = self.E.dot(self.W.T)

        # updates
        if self.batch > 1:
            np.subtract(self.W, self.optimizer(np.sum(np.einsum("bi,bj->bij", self.X, self.learning_rate*self.E), axis=0)), self.W)
            # TODO : sharing bias to all batch
            self.bias -= np.sum(self.learning_rate * self.E)
        else:
            np.subtract(self.W, self.optimizer(np.outer(self.X, self.learning_rate * self.E)), self.W)
            if len(self.E) >= 2:
                self.bias -= np.sum(self.learning_rate * self.E, axis=1)
                #self.bias -= self.optimizer(np.sum(self.learning_rate * self.E, axis=1))
            else:
                # TODO : just workaround for shape = (1)
                self.bias -= np.sum(self.learning_rate * self.E)
                #self.bias -= self.optimizer(np.sum(self.learning_rate * self.E))

        if self.original_shape:
            err_delta = err_delta.reshape(self.original_shape)
        return err_delta