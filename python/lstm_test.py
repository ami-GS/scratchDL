from layer import FullyConnect, Conv2D, MaxPooling2D, LSTM
from activation import Sigmoid, ReLU, Tanh, Softmax
from optimizer import Momentum
from network import Network
from loss import MSE, CrossEntropy
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataNum = 100
    dataPoints = 2**14
    learning_rate = 0.01
    # X shold be varried
    dataX = np.zeros((dataNum, dataPoints))
    for i in range(dataNum):
        prefix = np.random.rand()*2
        dataX[i,:] = np.linspace(prefix, prefix+np.pi*4, dataPoints)
    labelY = np.sin(dataX)
    epoch = 60
    batchSize = 20
    input_shape = 2048
    units = 256
    last_units = 64
    lstm = Network([LSTM(units, input_shape),
                    FullyConnect(units=last_units, input_shape=units)
                ],
                   learning_rate=learning_rate,
                   optimizer=Momentum(0.9),
                   batch=batchSize,
    )
    
    err_prev = 0
    for e in range(epoch):
        err = 0
        for batchIdx in range(0, dataNum, batchSize):
            batchData = labelY[batchIdx:batchIdx + batchSize, :]
            for timeIdx in range(0, dataPoints-input_shape, input_shape):
                err += lstm.train(batchData[:, timeIdx:timeIdx+input_shape], batchData[:, timeIdx+input_shape:timeIdx+input_shape+last_units], loss=MSE())
        print "epoch", e
        print "\t", err/batchSize
        err_prev = err

    print lstm.predict(labelY[:batchSize, :input_shape]), labelY[:batchSize, input_shape:input_shape+units]
    nxt = lstm.predict(labelY[:batchSize,:input_shape])
    plt.plot(dataX[0,input_shape:input_shape+last_units], nxt[0, :], dataX[0, input_shape:input_shape+last_units], labelY[0, input_shape:input_shape+last_units])
    plt.show()