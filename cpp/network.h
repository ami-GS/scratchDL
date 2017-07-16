#ifndef CPP_NETWORK_H_
#define CPP_NETWORK_H_

#include "layer.h"
#include "loss.h"

class Network {
public:
    float* learning_rate;
    int layerNum;
    Layer** layers;
    Loss* loss;
    Network(int layerNum, Layer** layers, Loss* loss);
    ~Network();
    int configure(int batch, float learning_rate);
    void train(float* data, int* label);
};


#endif // CPP_NETWORK_H_
