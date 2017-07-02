#include "layer.h"
#include <stdlib.h>
#include <random>

Layer::Layer(int input_shape, int units) : batch(1), input_shape(input_shape), units(units), prevLayer(nullptr) {}
Layer::~Layer() {
    delete this->E;
    delete this->Y;
    delete this->X;
}

FullyConnect::FullyConnect(int input_shape, int units) : Layer(input_shape, units) {}
FullyConnect::~FullyConnect() {
    delete this->W;
}

int FullyConnect::configure(int batch, Layer* prevLayer) {
    this->batch = batch;
    if (prevLayer != nullptr) {
        this->prevLayer = prevLayer;
    }
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


Conv2D::Conv2D(int input_shape, int channel, int filter, int kernel_size, int stride, int padding) : Layer(input_shape, (input_shape + 2*padding - kernel_size)/stride + 1), i_rowcol((int)std::sqrt((float)input_shape)), channel(channel), filter(filter), kernel_size(kernel_size), stride(stride), padding(padding) {
    this->u_rowcol = (int)std::sqrt((float)this->units);
}
Conv2D::~Conv2D() {
    delete this->F;
}

int Conv2D::configure(int batch, Layer* prevLayer) {
    if (prevLayer != nullptr) {
        this->prevLayer = prevLayer;
    }
    this->batch = batch;
    if (this->stride <= 0) {
        // warning
        this->stride = 1;
    }
    this->X = (float*)malloc(sizeof(float)*this->batch*this->channel*this->input_shape);
    // for filter and data matmul
    //this->X = (float*)malloc(sizeof(float)*this->batch*this->channel*this->kernel_size*this->kernel_size*this->units*this->units);
    this->Y = (float*)malloc(sizeof(float)*this->batch*this->filter*this->units);
    this->E = (float*)malloc(sizeof(float)*this->batch*this->channel*this->input_shape);
    this->F = (float*)malloc(sizeof(float)*this->filter*this->kernel_size*this->kernel_size);
    return 1;
}

void Conv2D::forward(float* x) {
    for (int i = 0; i < this->batch*this->filter*this->units; i++) {
        this->Y[i] = 0;
    }
    for (int b = 0; b < this->batch; b++) {
        for (int c = 0; c < this->channel; c++) {
            for (int f = 0; f < this->filter; f++) {
                for (int ro = 0; ro + this->kernel_size < this->i_rowcol; ro += this->stride) {
                    for (int co = 0; co + this->kernel_size < this->i_rowcol; co += this->stride) {
                        for (int ki = 0; ki < this->kernel_size; ki++) {
                            for (int kj = 0; kj < this->kernel_size; kj++) {
                                this->Y[b*this->units*this->filter+f*this->units+ro*this->i_rowcol+co] += this->X[b*this->input_shape*this->channel+c*this->input_shape+ro*this->i_rowcol+co+ki*this->i_rowcol+kj] * this->F[ki*this->kernel_size+kj];
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}


void Conv2D::backward(float* e) {
    for (int i = 0; i < this->batch*this->channel*this->input_shape; i++) {
        this->E[i] = 0;
    }
    for (int b = 0; b < this->batch; b++) {
        for (int c = 0; c < this->channel; c++) {
            for (int f = 0; f < this->filter; f++) {
                for (int ro = 0; ro < this->u_rowcol; ro++) {
                    for (int co = 0; co < this->u_rowcol; co++) {
                        for (int ki = 0; ki < this->kernel_size; ki++) {
                            for (int kj = 0; kj < this->kernel_size; kj++) {
                                this->E[b*this->channel*this->input_shape+c*this->input_shape+ro*this->stride+co*this->stride+ki*this->input_shape+kj] += this->Y[b*this->filter*this->units+f*this->units+ro*this->u_rowcol+co] * this->F[ki*this->kernel_size*kj];
                            }
                        }
                    }
                }
            }
        }
    }

    for (int b = 0; b < this->batch; b++) {
        for (int c = 0; c < this->channel; c++) {
            for (int f = 0; f < this->filter; f++) {

            }
        }
    }
}
