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
    #pragma omp  parallel for
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->Y[bi] = 0;
        this->E[bi] = 0;
    }
    return 1;
}

Sigmoid::Sigmoid() {}
Sigmoid::~Sigmoid() {}

void Sigmoid::forward(float* x) {
    #pragma omp  parallel for
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->Y[bi] = 1/(1+std::exp(-x[bi]));
    }
    return;
}


void Sigmoid::backward(float* e) {
    #pragma omp  parallel for
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->E[bi] = e[bi] * this->Y[bi]*(1-this->Y[bi]);
    }
    return;
}


ReLU::ReLU() {}
ReLU::~ReLU() {}

void ReLU::forward(float* x) {
    #pragma omp  parallel for
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->Y[bi] = (x[bi] > 0) * x[bi];
    }
    return;
}


void ReLU::backward(float* e) {
    #pragma omp  parallel for
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->E[bi] = (this->Y[bi] > 0) * e[bi];
    }
    return;
}


Softmax::Softmax() {}
Softmax::~Softmax() {}

void Softmax::forward(float* x) {
    float* maxVal = (float*)malloc(sizeof(float)*this->batch);
    for (int b = 0; b < this->batch; b++) {
        maxVal[b] = std::abs(x[b*this->input_shape]);
        for (int i = 1; i < this->input_shape; i++) {
            if (maxVal[b] < std::abs(x[b*this->input_shape+i])) {
                maxVal[b] = std::abs(x[b*this->input_shape+i]);
            }
        }
    }

    float tmp;
    for (int b = 0; b < this->batch; b++) {
        tmp = 0;
        for (int i = 0; i < this->input_shape; i++) {
            this->Y[b*this->input_shape+i] = std::exp(x[b*this->input_shape+i]/maxVal[b]);
            tmp += this->Y[b*this->input_shape+i];
        }
        for (int i = 0; i < this->input_shape; i++) {
            this->Y[b*this->input_shape+i] = this->Y[b*this->input_shape+i] / tmp;
        }
    }
    return;
}

void Softmax::backward(float* e) {
    this->E = e;
    return;
}
