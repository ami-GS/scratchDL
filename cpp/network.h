#ifndef CPP_NETWORK_H_
#define CPP_NETWORK_H_

#include "layer.h"
#include "loss.h"

class Network {
public:
    float* learning_rate;
    int layerNum;
    vector<Layer*>* layers;
    Loss* loss;
    Network(int layerNum, Loss* loss, vector<Layer*>* vLayers);
    Network(Loss* loss, vector<Layer*>* vLayers);
    ~Network();
    int configure(int batch, float learning_rate, float v_param, phase_t phase);
    void train(vector<float> *data, int* label);
};


#endif // CPP_NETWORK_H_
