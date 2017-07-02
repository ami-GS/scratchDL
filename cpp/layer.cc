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
    this->batch = batch;
    this->Y = (float*)malloc(sizeof(float)*this->batch*this->units);
    this->W = (float*)malloc(sizeof(float)*this->input_shape*this->units);
    this->E = (float*)malloc(sizeof(float)*this->batch*this->input_shape);
    this->X = (float*)malloc(sizeof(float)*this->batch*this->input_shape);
    for (int i = 0; i < this->batch*this->units; i++) {
            this->Y[i] = 0;
    }
    for (int i = 0; i < this->batch*this->input_shape; i++) {
            this->E[i] = 0;
            this->X[i] = 0;
    }
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> rand(-1.0,1.0);
    for (int i = 0; i < this->input_shape; i++) {
        for (int u = 0; u < this->units; u++) {
            this->W[i*this->units + u] = rand(mt);
        }
    }
    return 1;
}

void FullyConnect::forward(float* x) {
    for (int i = 0; i < this->batch*this->units; i++) {
        this->Y[i] = 0;
    }
    this->X = x;
    for (int b = 0; b < this->batch; b++) {
        for (int i = 0; i < this->input_shape; i++) {
            for (int u = 0; u < this->units; u++) {
                this->Y[b*this->units + u] += x[b*this->input_shape + i] * this->W[i*this->units + u];
            }
        }
    }
    return;
}

void FullyConnect::backward(float* e) {
    for (int i = 0; i < this->batch*this->input_shape; i++) {
        this->E[i] = 0;
    }
    for (int b = 0; b < this->batch; b++) {
        for (int u = 0; u < this->units; u++) {
            for (int i = 0; i < this->input_shape; i++) {
                this->E[b*this->input_shape + i] += e[b*this->units + u] * this->W[i*this->units+u];
            }
        }
    }

    for (int b = 0; b < this->batch; b++) {
        for (int i = 0; i < this->input_shape; i++) {
            for (int u = 0; u < this->units; u++) {
                // TODO : not good for division in each calc
                this->W[i*this->units + u] -= (this->X[b*this->input_shape + i] * e[b*this->units + u])/this->batch;
            }
        }
    }

    return;
}
