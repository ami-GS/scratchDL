from layer import FullyConnect, Conv2D, MaxPooling2D
from activation import Sigmoid, ReLU, Tanh, Softmax
from optimizer import Momentum
from network import Network
from loss import MSE, CrossEntropy
import numpy as np

if __name__ == "__main__":
    dataNum = 5000
    learning_rate = 0.001

    # 12 * 12 image
    input_shape = 144
    epoch = 10
    batchSize = 50
    channel = 1
    last_units = 2

    dataset = np.zeros((dataNum, channel, input_shape)) # 1 for channel
    label = np.zeros((dataNum, last_units))
    for i in range(last_units):
        # currently -1~0. 0~1 can be used due to some issue
        dataset[dataNum/last_units*i:dataNum/last_units*(i+1),:,:] = np.random.uniform(i-1, i, (dataNum/last_units,channel,input_shape))
        label[dataNum/last_units*i:dataNum/last_units*(i+1),i] = 1
    net = Network(
        [Conv2D(4, 3, input_shape=input_shape), # 1*12*12  -> 4*10*10
         ReLU(),
         MaxPooling2D(2), # 10*10*4 -> 9*9*4
         FullyConnect(units=last_units, input_shape=324),
         Softmax(),
     ],
    #FullyConnect(units=last_units)],
        # FullyConnect(units=last_units, input_shape=144)],
        learning_rate = learning_rate,
        optimizer=Momentum(0.9),
        batch=batchSize)

    err_prev = 0
    for e in range(epoch):
        err = 0
        for batchIdx in range(0, dataNum, batchSize):
            batchData = dataset[batchIdx:batchIdx + batchSize,:,:]
            batchLabel = label[batchIdx:batchIdx + batchSize,:]
            err += net.train(batchData, batchLabel, loss=MSE())
        print "epoch", e
        print "\t", err
        if abs(err_prev - err) < 10e-15:
            print "early_stop"
            break
        err_prev = err

    print net.predict(dataset[0, :, :].reshape(1,1,input_shape)), label[0,:]
    print net.predict(dataset[dataNum-1, :].reshape(1,1,input_shape)), label[dataNum-1,:]

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
    print net.predict(dataset[0, :], label[0,:]
    print net.predict(dataset[dataNum-1, :], label[dataNum-1,:]
    #ans = net.predict(dataset[:,:], batchS.ize)
    """