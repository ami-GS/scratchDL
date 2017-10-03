#include "loss.h"
#include <cmath>
#include <stdlib.h>

Loss::Loss() {}
Loss::~Loss() {
    this->D.clear();
    this->D.shrink_to_fit();
}

int Loss::configure(int batch, Layer* prevLayer) {
    if (prevLayer != nullptr) {
        this->prevLayer = prevLayer;
    }
    this->batch = batch;
    this->batch_inv = 1/batch;
    this->D.resize(this->batch*this->prevLayer->units);
    return 1;
}

MSE::MSE() {}
MSE::~MSE() {}

int MSE::configure(int batch, Layer* prevLayer) {
    return Loss::configure(batch, prevLayer);
}

float MSE::error(vector<float> *x, int* label) {
    float e = 0;
    #pragma omp  parallel for reduction(+:e)
    for (int i = 0; i < this->batch*this->prevLayer->units; i++) {
        e += std::pow(x->at(i)-(float)label[i], 2)*0.5*this->batch_inv;
    }
    return e;
}

void MSE::partial_derivative(vector<float>* x, int* label) {
    #pragma omp  parallel for
    for (int i = 0; i < this->batch*this->prevLayer->units; i++) {
        this->D[i] = - ((float)label[i] - x->at(i));
    }
    return;
}
