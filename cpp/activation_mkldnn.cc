#include "activation_mkldnn.h"
#include <cmath>
#include <iostream>

Activation::Activation() : Layer(0, 0) {}
Activation::~Activation() {}

int Activation::configure(int batch, float learning_rate, Layer* prevLayer, mkldnn::engine backend) {
    return 1;
}

ReLU::ReLU() {}
ReLU::~ReLU() {}


Softmax::Softmax() {}
Softmax::~Softmax() {}
