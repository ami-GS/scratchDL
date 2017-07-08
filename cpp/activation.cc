#include "activation.h"
#include <cmath>

Activation::Activation() : Layer(0, 0) {}
Activation::~Activation() {}

int Activation::configure(int batch, float learning_rate, Layer* prevLayer) {
    this->input_shape = prevLayer->units;
    if (prevLayer->channel != 0) {
        this->input_shape = prevLayer->units*prevLayer->channel;
    }
    this->learning_rate = learning_rate;
    this->units = this->input_shape;
    this->batch = batch;
    this->Y = (float*)malloc(sizeof(float)*this->batch*this->input_shape);
    this->E = (float*)malloc(sizeof(float)*this->batch*this->input_shape);
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->Y[bi] = 0;
        this->E[bi] = 0;
    }
    return 1;
}

Sigmoid::Sigmoid() {}
Sigmoid::~Sigmoid() {}

void Sigmoid::forward(float* x) {
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->Y[bi] = 1/(1+std::exp(-x[bi]));
    }
    return;
}


void Sigmoid::backward(float* e) {
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->E[bi] = e[bi] * this->Y[bi]*(1-this->Y[bi]);
    }
    return;
}


ReLU::ReLU() {}
ReLU::~ReLU() {}

void ReLU::forward(float* x) {
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->Y[bi] = (x[bi] > 0) * x[bi];
    }
    return;
}


void ReLU::backward(float* e) {
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->E[bi] = (this->Y[bi] > 0) * e[bi];
    }
    return;
}
