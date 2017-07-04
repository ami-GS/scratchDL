#include "activation.h"
#include <cmath>

Activation::Activation() : Layer(0, 0) {}
Activation::~Activation() {}

int Activation::configure(int batch, Layer* prevLayer) {
    this->batch = batch;
    this->Y = (float*)malloc(sizeof(float)*this->batch*this->input_shape);
    this->E = (float*)malloc(sizeof(float)*this->batch*this->input_shape);
    for (int b = 0; b < this->batch; b++) {
        for (int i = 0; i < this->input_shape; i++) {
            this->Y[b*this->input_shape+i] = 0;
            this->E[b*this->input_shape+i] = 0;
        }
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
    for (int bi = 0; bi < this->batch; bi++) {
        this->E[bi] = e[bi] * this->Y[bi]*(1-this->Y[bi]);
    }
    return;
}
