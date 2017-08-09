#include "network.h"
#include "loss.h"
#include "layer.h"


Network::Network(int layerNum, Layer** layers, Loss* loss) : layerNum(layerNum), layers(layers), loss(loss) {
}
Network::~Network() {
    for (int i = 0; i < this->layerNum; i++) {
        delete this->layers[i];
    }
};


int Network::configure(int batch, float learning_rate, float v_param, phase_t phase) {
    Layer* prevLayer = nullptr;
    for (int i = 0; i < this->layerNum; i++) {
        this->layers[i]->configure(batch, learning_rate, v_param, prevLayer, phase);
        prevLayer = this->layers[i];
    }
    loss->configure(batch, prevLayer);
    return 1;
}

void Network::train(float* data, int* label) {
    for (int j = 0; j < this->layerNum; j++) {
        this->layers[j]->forward(data);
        data = this->layers[j]->Y;
    }
    this->loss->partial_derivative(data, label);
    float* e = this->loss->D;
    for (int j = this->layerNum-1; j >= 0; j--) {
        this->layers[j]->backward(e);
        e = this->layers[j]->E;
    }
    return;
}
