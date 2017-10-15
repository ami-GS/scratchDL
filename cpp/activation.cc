#include "activation.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iterator>


Activation::Activation() : Layer(0, 0) {}
Activation::~Activation() {}

int Activation::configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase) {
    this->input_shape = prevLayer->units;
    if (prevLayer->channel != 0) {
        this->input_shape *= prevLayer->channel;
    }
    this->units = this->input_shape;
    Layer::configure(batch, learning_rate, v_param, prevLayer, phase);
    this->Y.resize(this->batch*this->input_shape);
    if (this->phase == TRAIN) {
        this->E.resize(this->batch*this->input_shape);
    }
    return 1;
}

Sigmoid::Sigmoid() {}
Sigmoid::~Sigmoid() {}

void Sigmoid::forward(vector<float> *x) {
    #pragma omp  parallel for
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->Y[bi] = 1/(1+std::exp(-x->at(bi)));
    }
    return;
}


void Sigmoid::backward(vector<float> *e) {
    #pragma omp  parallel for
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->E[bi] = e->at(bi) * this->Y[bi]*(1-this->Y[bi]);
    }
    return;
}


ReLU::ReLU() {}
ReLU::~ReLU() {}

void ReLU::forward(vector<float> *x) {
    #pragma omp  parallel for
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->Y[bi] = (x->at(bi) > 0) * x->at(bi);
    }
    return;
}


void ReLU::backward(vector<float> *e) {
    #pragma omp  parallel for
    for (int bi = 0; bi < this->batch*this->input_shape; bi++) {
        this->E[bi] = (this->Y[bi] > 0) * e->at(bi);
    }
    return;
}


Softmax::Softmax() {}
Softmax::~Softmax() {
    delete this->maxVal;
}

void Softmax::forward(vector<float> *x) {
    if (this->maxVal == nullptr) {
        this->maxVal = (float*)malloc(sizeof(float)*this->batch);
    }
    #pragma omp parallel for
    for (int b = 0; b < this->batch; b++) {
        maxVal[b] = std::abs(x->at(b*this->input_shape));
        for (int i = 1; i < this->input_shape; i++) {
            if (this->maxVal[b] < std::abs(x->at(b*this->input_shape+i))) {
                this->maxVal[b] = std::abs(x->at(b*this->input_shape+i));
            }
        }
    }

    float tmp;
    #pragma omp parallel for private(tmp)
    for (int b = 0; b < this->batch; b++) {
        tmp = 0;
        for (int i = 0; i < this->input_shape; i++) {
            this->Y[b*this->input_shape+i] = std::exp(x->at(b*this->input_shape+i)/this->maxVal[b]);
            tmp += this->Y[b*this->input_shape+i];
        }
        for (int i = 0; i < this->input_shape; i++) {
            this->Y[b*this->input_shape+i] = this->Y[b*this->input_shape+i] / tmp;
        }
    }
    return;
}

void Softmax::backward(vector<float> *e) {
    copy(e->begin(), e->end(), back_inserter(this->E));
    return;
}
