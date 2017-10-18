#include "network.h"
#include "loss.h"
#include "layer.h"


Network::Network(int layerNum, Loss* loss, vector<Layer*>* layers) : layerNum(layerNum), layers(layers), loss(loss) {
}

Network::Network(Loss* loss, vector<Layer*>* layers) : layers(layers), loss(loss) {
    this->layerNum = layers->size();
}

Network::~Network() {
    for (int i = 0; i < this->layerNum; i++) {
        delete this->layers->at(i);
    }
};


int Network::configure(int batch, float learning_rate, float v_param, phase_t phase) {
    Layer* prevLayer = nullptr;
    for (int i = 0; i < this->layerNum; i++) {
        this->layers->at(i)->configure(batch, learning_rate, v_param, prevLayer, phase);
        prevLayer = this->layers->at(i);
    }
    loss->configure(batch, prevLayer);
    return 1;
}

void Network::train(vector<float> *data, int* label) {
    for (int j = 0; j < this->layerNum; j++) {
        this->layers->at(j)->forward(data);
        data = &this->layers->at(j)->Y;
    }
    this->loss->partial_derivative(data, label);
    vector<float> *e = &this->loss->D;
    for (int j = this->layerNum-1; j >= 0; j--) {
        this->layers->at(j)->backward(e);
        e = &this->layers->at(j)->E;
    }
    return;
}

float Network::getLossError(int* label) {
    return this->loss->error(&this->layers->back()->Y, label);
}
