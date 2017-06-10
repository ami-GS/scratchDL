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
        self.channel = x.shape[1]

        if not self.x_rowcol:
            self.x_rowcol = int(np.sqrt(self.input_shape))
            self.y_rowcol = self.x_rowcol - self.kernel_size+1

        self.X = np.reshape(self.X, (self.batch, self.channel, self.x_rowcol, self.x_rowcol))
        self.Y = np.zeros((self.batch, self.filterNum, self.y_rowcol, self.y_rowcol))

        for yi in range(0, self.y_rowcol, self.strides[0]):
            for yj in range(0, self.y_rowcol, self.strides[1]):
                self.Y[:,:,yi,yj] = np.sum(np.einsum("bcij,fij->bcfij", self.X[:,:,yi:yi+self.kernel_size,yj:yj+self.kernel_size], self.filters[:,:,:]), axis=(1,3,4))
        return self.Y

    def backward(self, err_delta):
        self.E = err_delta
        err_delta = np.zeros((self.batch, self.channel, self.x_rowcol, self.x_rowcol))
        # TODO : temporally using batch
        for b in range(self.batch):
            for yi in range(self.y_rowcol):
                for yj in range(self.y_rowcol):
                    err_delta[b, :, yi:yi+self.kernel_size, yj:yj+self.kernel_size] = np.add(
                        err_delta[b, :, yi:yi+self.kernel_size, yj:yj+self.kernel_size],
                        np.sum((self.E[b,:,yi,yj] * self.filters.T).T, axis=0))

        for yi in range(0, self.y_rowcol, self.strides[0]):
            for yj in range(0, self.y_rowcol, self.strides[1]):
                self.filters -= np.sum(np.sum(
                    np.einsum("bcij,bf->bcfij", self.X[:,:,yi:yi+self.kernel_size,yj:yj+self.kernel_size], self.E[:,:,yi,yj]),
                    axis=1), axis=0)/self.batch #sum channel, then batch

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

        self.channel = x.shape[1]
        self.maxLocs = np.zeros((self.batch, self.channel, self.y_rowcol, self.y_rowcol, 2), dtype=int)
        # TODO: workaround
        self.Y = np.zeros((self.batch, self.channel, self.y_rowcol, self.y_rowcol))

        for c in range(self.channel):
            for yi in range(0, self.y_rowcol, self.strides[0]):
                for yj in range(0, self.y_rowcol, self.strides[1]):
                    tmp = self.X[:, c, yi:yi+self.kernel_size, yj:yj+self.kernel_size].reshape(self.batch, -1).argmax(1)
                    self.maxLocs[:, c, yi, yj, 0] = (tmp/self.kernel_size).astype(int)
                    self.maxLocs[:, c, yi, yj, 1] = tmp%self.kernel_size
                    self.Y[:, c, yi, yj] = np.max(self.X[:, c, yi:yi+self.kernel_size, yj:yj+self.kernel_size], axis=(1,2))
        return self.Y

    def backward(self, err_delta):
        self.E = err_delta
        err_delta = np.zeros((self.batch, self.channel, self.x_rowcol, self.x_rowcol))

        for b in range(self.batch):
            for c in range(self.channel):
                for yi in range(self.y_rowcol):
                    for yj in range(self.y_rowcol):
                        # TODO : really bad way
                        err_delta[b,c,yi+self.maxLocs[b,c,yi,yj,0], yj+self.maxLocs[b,c,yi,yj,1]] += self.E[b,c,yi,yj]

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
        if len(x.shape) > 1:
            self.original_shape = x.shape
            x = x.reshape(self.original_shape[0], reduce(lambda x,y:x*y, self.original_shape[1:]))
        self.X = x
        self.Y = x.dot(self.W) + self.bias
        return self.Y

    def backward(self, err_delta):
        self.E = err_delta
        err_delta = self.E.dot(self.W.T)

        # updates
        np.subtract(self.W, self.optimizer(np.sum(np.einsum("bi,bj->bij", self.X, self.learning_rate*self.E), axis=0))/self.batch, self.W)
        self.bias -= np.sum(self.learning_rate * self.E)

        if self.original_shape:
            err_delta = err_delta.reshape(self.original_shape)
        return err_delta