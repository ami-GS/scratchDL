from layer import FullyConnect, Conv2D, MaxPooling2D
from activation import Sigmoid, ReLU, Tanh, Softmax
from optimizer import Momentum
from network import Network
from loss import MSE
import numpy as np

if __name__ == "__main__":
    dataNum = 5000
    learning_rate = 0.001

    # 8*8 image
    input_shape = 64
    epoch = 10
    last_units = 1
    dataset = np.zeros((dataNum, input_shape))
    dataset[:dataNum/2,:] = np.random.uniform(-1, 0, (dataNum/2,input_shape))
    dataset[dataNum/2:,:] = np.random.uniform(0, 1, (dataNum/2,input_shape))

    label = np.zeros((dataNum, last_units))
    label[:dataNum/2,:] = np.ones((dataNum/2, last_units))

    net = Network(
        [Conv2D(4, 1, 3, input_shape=input_shape), # 64 * 4 -> 36*4
         ReLU(),
         MaxPooling2D(2), # 6*6*4 -> 5*5*4
         FullyConnect(units=last_units, input_shape=100)],
         #FullyConnect(units=last_units)],
        # FullyConnect(units=last_units, input_shape=144)],
        learning_rate = learning_rate,
        optimizer=Momentum(0.02))
    for e in range(epoch):
        print e
        for i in range(dataNum):
            net.train(dataset[i], label[i], loss=MSE())
        

    """
    # 100 epoch 10 batch  8*8 image
    epoch = 100
    batchSize = 10

    input_shape = 4
    last_units = 2
    dataset = np.zeros((dataNum, input_shape))
    dataset[:dataNum/2,:] = np.random.uniform(-1, 0, (dataNum/2,input_shape))
    dataset[dataNum/2:,:] = np.random.uniform(0, 1, (dataNum/2,input_shape))
    
    label = np.zeros((dataNum, last_units))
    label[:dataNum/2,:] = np.ones((dataNum/2, last_units))

    net = Network(
        [FullyConnect(input_shape*2, input_shape),
         Sigmoid(),
         FullyConnect(input_shape/2, input_shape*2),
         ReLU(),
         FullyConnect(units=last_units, input_shape=input_shape/2)],
        #Softmax()],
        batch = batchSize,
        learning_rate = learning_rate,
        optimizer=Momentum(0.02))
    
    for e in range(epoch):
        err = 0
        for batchIdx in range(0, dataNum, batchSize):
            batchData = dataset[batchIdx:batchIdx + batchSize, :]
            batchLabel = label[batchIdx:batchIdx + batchSize, :]
            err += net.train(batchData, batchLabel, loss=MSE())
        if e == 0 or (e+1)%100 == 0:
            print "epoch", e+1
            print "\t", err
    print dataset.shape, label.shape
    """

    print net.predict(dataset[0, :]), label[0,:]
    print net.predict(dataset[dataNum-1, :]), label[dataNum-1,:]
    #ans = net.predict(dataset[:,:], batchSize)