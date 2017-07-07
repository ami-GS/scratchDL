#include "loss.h"
#include <cmath>
#include <stdlib.h>

Loss::Loss() {}
Loss::~Loss() {
    delete this->D;
}

MSE::MSE() {}
MSE::~MSE() {}

int MSE::configure(int batch, Layer* prevLayer) {
    if (prevLayer != nullptr) {
        this->prevLayer = prevLayer;
    }
    this->batch = batch;
    this->D = (float*)malloc(sizeof(float)*this->batch*this->prevLayer->units);
    return 1;
}

float MSE::error(float* x, int* label) {
    float e = 0;
    float tmp = 0;
    for (int b = 0; b < this->batch; b++) {
        tmp = 0;
        for (int i = 0; i < this->prevLayer->units; i++) {
            tmp += std::pow(std::abs(x[i]-(float)label[i]), 2)*0.5;
        }
        e += tmp/this->batch;
    }
    return e;
}

void MSE::partial_derivative(float* x, int* label) {
    for (int i = 0; i < this->batch*this->prevLayer->units; i++) {
        this->D[i] = - ((float)label[i] - x[i]);
    }
    return;
}
