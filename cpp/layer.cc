#include "layer.h"
#include <stdlib.h>
#include <random>

Layer::Layer(int input_shape, int units) : batch(1), input_shape(input_shape), units(units), prevLayer(nullptr) {}
Layer::~Layer() {}

FullyConnect::FullyConnect(int input_shape, int units) : Layer(input_shape, units) {}
FullyConnect::~FullyConnect() {
    delete this->E;
    delete this->W;
    delete this->Y;
    delete this->X;
}

int FullyConnect::configure(int batch, Layer* prevLayer) {
    return 1;
}

void FullyConnect::forward(float* x) {
    return;
}

void FullyConnect::backward(float* e) {
    return;
}
