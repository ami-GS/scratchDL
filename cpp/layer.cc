#include "layer.h"
#include "mkldnn.hpp"
#include <stdlib.h>
#include <random>

Layer::Layer(int input_shape, int units) : batch(1), filter(0), channel(0), input_shape(input_shape), units(units), prevLayer(nullptr) {}
Layer::~Layer() {
    free(this->E);
    free(this->Y);
    free(this->X);
}

FullyConnect::FullyConnect(int input_shape, int units) : Layer(input_shape, units) {}
FullyConnect::~FullyConnect() {
    free(this->W);
}

int FullyConnect::configure(int batch, float learning_rate, Layer* prevLayer) {
    this->batch = batch;
    this->learning_rate = learning_rate;
    if (prevLayer != nullptr) {
        this->prevLayer = prevLayer;
    }
    this->Y = (float*)malloc(sizeof(float)*this->batch*this->units);
    this->W = (float*)malloc(sizeof(float)*this->input_shape*this->units);
    this->B = (float*)malloc(sizeof(float));
    this->E = (float*)malloc(sizeof(float)*this->batch*this->input_shape);
    this->X = (float*)malloc(sizeof(float)*this->batch*this->input_shape);
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> rand(-1.0,1.0);
    #pragma omp  parallel for
    for (int iu = 0; iu < this->input_shape*this->units; iu++) {
            this->W[iu] = rand(mt);
    }
    *this->B = rand(mt);
    return 1;
}

    mkldnn::memory::dims fc_src = {batch, this->input_shape};
    mkldnn::memory::dims fc_dst = {batch, this->units};
    mkldnn::memory::dims fc_weights = {this->input_shape, this->units};

int FullyConnect::configure_mkldnn(int batch, float learning_rate, Layer* prevLayer, mkldnn::engine backend) {
    this->batch = batch;
    this->learning_rate = learning_rate;
    if (prevLayer != nullptr) {
        prevLayer->nxtLayer = this;
        this->prevLayer = prevLayer;
    }

    auto fc_src_md = mkldnn::memory::desc({fc_src}, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
    auto fc_dst_md = mkldnn::memory::desc({fc_dst}, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
    auto fc_weights_md = mkldnn::memory::desc({fc_weights}, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
    auto fc_desc = mkldnn::inner_product_forward::desc(mkldnn::prop_kind::forward, fc_src_md, fc_weights_md, fc_dst_md);
    auto fc_prim_desc = mkldnn::inner_product_forward::primitive_desc(fc_desc, backend);

    return 1;
}

void FullyConnect::forward(float* x) {
    #pragma omp  parallel for
    for (int i = 0; i < this->batch*this->units; i++) {
        this->Y[i] = 0;
    }
    this->X = x;
    #pragma omp  parallel for
    for (int b = 0; b < this->batch; b++) {
        for (int i = 0; i < this->input_shape; i++) {
            for (int u = 0; u < this->units; u++) {
                this->Y[b*this->units + u] += x[b*this->input_shape + i] * this->W[i*this->units + u] + *this->B;
            }
        }
    }
    return;
}

void FullyConnect::backward(float* e) {
    #pragma omp  parallel for
    for (int i = 0; i < this->batch*this->input_shape; i++) {
        this->E[i] = 0;
    }
    #pragma omp  parallel for
    for (int b = 0; b < this->batch; b++) {
        for (int u = 0; u < this->units; u++) {
            for (int i = 0; i < this->input_shape; i++) {
                this->E[b*this->input_shape + i] += e[b*this->units + u] * this->W[i*this->units+u];
            }
        }
    }

    #pragma omp  parallel for
    for (int b = 0; b < this->batch; b++) {
        for (int i = 0; i < this->input_shape; i++) {
            for (int u = 0; u < this->units; u++) {
                // TODO : not good for division in each calc
                this->W[i*this->units + u] -= (this->learning_rate * this->X[b*this->input_shape + i] * e[b*this->units + u])/this->batch;
            }
        }
    }
    #pragma omp  parallel for
    for (int b = 0; b < this->batch; b++) {
        for (int u = 0; u < this->units; u++) {
            *this->B -= this->learning_rate * e[b*this->units + u]/this->batch;
        }
    }

    return;
}


Conv2D::Conv2D(int input_shape, int channel, int filter, int kernel_size, int stride, int padding) : Layer(input_shape, 0), i_rowcol((int)std::sqrt((float)input_shape)), kernel_size(kernel_size), stride(stride), padding(padding) {
    this->u_rowcol = (this->i_rowcol + 2*padding - kernel_size)/stride + 1;
    this->units = this->u_rowcol*this->u_rowcol;
    this->channel = channel;
    this->filter = filter;
}
Conv2D::~Conv2D() {
    free(this->F);
}

int Conv2D::configure(int batch, float learning_rate, Layer* prevLayer) {
    if (prevLayer != nullptr) {
        this->prevLayer = prevLayer;
    }
    this->batch = batch;
    this->learning_rate = learning_rate;
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

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> rand(-1.0,1.0);
    #pragma omp  parallel for
    for (int i = 0; i < this->filter*this->kernel_size*this->kernel_size; i++) {
        this->F[i] = rand(mt);
    }

int Conv2D::configure_mkldnn(int batch, float learning_rate, Layer* prevLayer, mkldnn::engine backend) {

    return 1;
}

void Conv2D::forward(float* x) {
    #pragma omp  parallel for
    for (int i = 0; i < this->batch*this->filter*this->units; i++) {
        this->Y[i] = 0;
    }
    this->X = x;
    #pragma omp  parallel for
    for (int b = 0; b < this->batch; b++) {
        for (int c = 0; c < this->channel; c++) {
            for (int f = 0; f < this->filter; f++) {
                for (int ro = 0; ro + this->kernel_size < this->i_rowcol; ro += this->stride) {
                    for (int co = 0; co + this->kernel_size < this->i_rowcol; co += this->stride) {
                        for (int ki = 0; ki < this->kernel_size; ki++) {
                            for (int kj = 0; kj < this->kernel_size; kj++) {
                                this->Y[b*this->units*this->filter+f*this->units+ro*this->i_rowcol+co] += this->X[b*this->input_shape*this->channel+c*this->input_shape+ro*this->i_rowcol+co+ki*this->i_rowcol+kj] * this->F[f*this->kernel_size*this->kernel_size+ki*this->kernel_size+kj];
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
    #pragma omp  parallel for
    for (int i = 0; i < this->batch*this->channel*this->input_shape; i++) {
        this->E[i] = 0;
    }
    #pragma omp  parallel for
    for (int b = 0; b < this->batch; b++) {
        for (int c = 0; c < this->channel; c++) {
            for (int f = 0; f < this->filter; f++) {
                for (int ro = 0; ro < this->u_rowcol; ro++) {
                    for (int co = 0; co < this->u_rowcol; co++) {
                        for (int ki = 0; ki < this->kernel_size; ki++) {
                            for (int kj = 0; kj < this->kernel_size; kj++) {
                                this->E[b*this->channel*this->input_shape+c*this->input_shape+ro*this->stride+co*this->stride+ki*this->i_rowcol+kj] += this->Y[b*this->filter*this->units+f*this->units+ro*this->u_rowcol+co] * this->F[f*this->kernel_size*this->kernel_size+ki*this->kernel_size+kj];
                            }
                        }
                    }
                }
            }
        }
    }

    #pragma omp  parallel for
    for (int b = 0; b < this->batch; b++) {
        for (int c = 0; c < this->channel; c++) {
            for (int f = 0; f < this->filter; f++) {
                for (int ro = 0; ro < this->u_rowcol; ro++) {
                    for (int co = 0; co < this->u_rowcol; co++) {
                        for (int ki = 0; ki < this->kernel_size; ki++) {
                            for (int kj = 0; kj < this->kernel_size; kj++) {
                                this->F[f*this->kernel_size*this->kernel_size+ki*this->kernel_size+kj] -= (this->learning_rate * this->X[b*this->channel*this->input_shape+c*this->input_shape+ro*this->i_rowcol+co+ki*this->i_rowcol+kj] * e[b*this->filter*this->units+f*this->units+ro*this->u_rowcol+co])/this->batch;
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}


MaxPooling2D::MaxPooling2D(int input_shape, int channel, int kernel_size, int stride) : Layer(input_shape, 0), i_rowcol((int)std::sqrt((float)input_shape)), kernel_size(kernel_size), stride(stride) {
    this->u_rowcol = (this->i_rowcol - kernel_size)/stride + 1;
    this->units = this->u_rowcol*this->u_rowcol;
    this->channel = channel;
}
MaxPooling2D::~MaxPooling2D() {
    free(this->L);
}

int MaxPooling2D::configure(int batch, float learning_rate, Layer* prevLayer) {
    if (prevLayer != nullptr) {
        this->prevLayer = prevLayer;
    }
    this->batch = batch;
    this->learning_rate = learning_rate;
    this->Y = (float*)malloc(sizeof(float)*this->batch*this->channel*this->units);
    this->L = (int*)malloc(sizeof(int)*this->batch*this->channel*this->units);
    this->E = (float*)malloc(sizeof(float)*this->batch*this->channel*this->input_shape);
    return 1;
}

int MaxPooling2D::configure_mkldnn(int batch, float learning_rate, Layer* prevLayer, mkldnn::engine backend) {
    return 1;
}

void MaxPooling2D::forward(float* x) {
    float tmp;
    #pragma omp  parallel for private(tmp)
    for (int b = 0; b < this->batch; b++) {
        for (int c = 0; c < this->channel; c++) {
            for (int ro = 0; ro < this->u_rowcol; ro++) {
                for (int co = 0; co < this->u_rowcol; co++) {
                    tmp = x[b*this->channel*this->input_shape+c*this->input_shape+ro*this->i_rowcol+co];
                    for (int ki = 0; ki < this->kernel_size; ki++) {
                        for (int kj = 1; kj < this->kernel_size; kj++) {
                            if (tmp < x[b*this->channel*this->input_shape+c*this->input_shape+ro*this->i_rowcol+co+ki*this->i_rowcol+kj]) {
                                tmp = x[b*this->channel*this->input_shape+c*this->input_shape+ro*this->i_rowcol+co+ki*this->i_rowcol+kj];
                                this->Y[b*this->channel*this->units+c*this->units+ro*this->u_rowcol+co] = tmp;
                                this->L[b*this->channel*this->units+c*this->units+ro*this->u_rowcol+co] = ki*this->i_rowcol+kj;
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}


void MaxPooling2D::backward(float* e) {
    #pragma omp  parallel for
    for (int i = 0; i < this->batch*this->channel*this->input_shape; i++) {
        this->E[i] = 0;
    }
    #pragma omp  parallel for
    for (int b = 0; b < this->batch; b++) {
        for (int c = 0; c < this->channel; c++) {
            for (int ro = 0; ro < this->u_rowcol; ro++) {
                for (int co = 0; co < this->u_rowcol; co++) {
                    this->E[b*this->input_shape*this->channel+c*this->input_shape+ro*this->i_rowcol+co+this->L[b*this->units*this->channel+c*this->units+ro*this->u_rowcol+co]] = e[b*this->units*this->channel+c*this->units+ro*this->u_rowcol+co];
                }
            }
        }
    }
    return;
}
