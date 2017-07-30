from layer import FullyConnect, Conv2D, MaxPooling2D, LSTM
from activation import Sigmoid, ReLU, Tanh, Softmax
from optimizer import Momentum
from network import Network
from loss import MSE, CrossEntropy
import numpy as np

if __name__ == "__main__":
    dataNum = 1
    dataPoints = 10**4
    learning_rate = 0.01
    oneRange = np.pi/dataPoints
    # X shold be varried
    dataX = np.linspace(0, np.pi*4, dataPoints)
    dataX = np.array([np.linspace(0, np.pi*4, dataPoints) for _ in range(dataNum)])
    labelY = np.sin(dataX)

    epoch = 1000
    batchSize = 1
    input_shape = 100
    units = 2
    lstm = Network([LSTM(units, input_shape)],
                   learning_rate=learning_rate,
                   optimizer=Momentum(0.9),
                   batch=batchSize,
    )
    
    err_prev = 0
    for e in range(epoch):
        err = 0
        for batchIdx in range(0, dataNum, batchSize):
            batchData = dataX[batchIdx:batchIdx + batchSize, :]
            batchLabel = labelY[batchIdx:batchIdx + batchSize, :]
            for timeIdx in range(0, dataPoints-input_shape, input_shape):
                err += lstm.train(batchData[:, timeIdx:timeIdx+input_shape], batchData[:, timeIdx+input_shape:timeIdx+input_shape+units], loss=MSE())
            #print batchIdx, err
        print "epoch", e
        print "\t", err/batchSize
        err_prev = err

    print lstm.predict(dataX[0:16, :100]), labelY[0:16, 100:102]
