#include "layer.h"
#include <stdlib.h>
#include <random>

Layer::Layer(int input_shape, int units) : batch(1), filter(0), channel(0), input_shape(input_shape), units(units), prevLayer(nullptr), momentum_a(1.0) {}
Layer::~Layer() {
    free(this->Y);
    if (this->phase == TRAIN) {
        free(this->E);
    }
}

int Layer::configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase) {
    this->batch = batch;
    this->batch_inv = 1/batch;
    this->learning_rate = learning_rate;
    this->momentum_a = v_param;
    this->phase = phase;
    if (prevLayer != nullptr) {
        prevLayer->nxtLayer = this;
        this->prevLayer = prevLayer;
    }
    return 1;
}

FullyConnect::FullyConnect(int input_shape, int units) : Layer(input_shape, units) {}
FullyConnect::~FullyConnect() {
    free(this->W);
    if (this->phase == TRAIN) {
        free(this->delta_buf);
    }
}

int FullyConnect::configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase) {
    Layer::configure(batch, learning_rate, v_param, prevLayer, phase);
    this->Y = (float*)malloc(sizeof(float)*this->batch*this->units);
    this->W = (float*)malloc(sizeof(float)*this->input_shape*this->units);
    this->B = (float*)malloc(sizeof(float));
    if (this->phase == TRAIN) {
        this->E = (float*)malloc(sizeof(float)*this->batch*this->input_shape);
        this->delta_buf = (float*)malloc(sizeof(float)*this->batch*this->units);
    }
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

void FullyConnect::forward(float* x) {
    #pragma omp  parallel for
    for (int i = 0; i < this->batch*this->units; i++) {
        this->Y[i] = 0;
    }
    #pragma omp  parallel for
    for (int b = 0; b < this->batch; b++) {
        for (int i = 0; i < this->input_shape; i++) {
            for (int u = 0; u < this->units; u++) {
                this->Y[b*this->units + u] += x[b*this->input_shape + i] * this->W[i*this->units + u] + *this->B;
            }
        }
    }
    if (this->phase == TRAIN) {
        this->X = x;
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

    // update
    if (this->momentum_a != 0) {
        #pragma omp  parallel for
        for (int b = 0; b < this->batch; b++) {
            for (int i = 0; i < this->input_shape; i++) {
                for (int u = 0; u < this->units; u++) {
                    this->W[i*this->units + u] -= this->momentum_a * this->delta_buf[b*this->units + u] + \
                        (this->learning_rate * this->X[b*this->input_shape + i] * e[b*this->units + u])*this->batch_inv;
                }
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < this->batch*this->units; i++) {
            this->delta_buf[i] = e[i];
        }
    } else {
        for (int b = 0; b < this->batch; b++) {
            for (int i = 0; i < this->input_shape; i++) {
                for (int u = 0; u < this->units; u++) {
                    this->W[i*this->units + u] -= this->learning_rate * this->X[b*this->input_shape + i] * e[b*this->units + u]*this->batch_inv;
                }
            }
        }
    }

    #pragma omp  parallel for
    for (int b = 0; b < this->batch; b++) {
        for (int u = 0; u < this->units; u++) {
            *this->B -= this->learning_rate * e[b*this->units + u]*this->batch_inv;
        }
    }

    return;
}


Conv2D::Conv2D(int input_shape, int channel, int filter, int kernel_size, int stride, int padding) : Layer(input_shape, 0), i_rowcol((int)std::sqrt((float)input_shape)), kernel_size(kernel_size), stride(stride), padding(padding) {
    if (stride <= 0) {
        // warning
        this->stride = 1;
    }
    this->u_rowcol = (this->i_rowcol + 2*padding - kernel_size)/stride + 1;
    this->units = this->u_rowcol*this->u_rowcol;
    this->channel = channel;
    this->filter = filter;
}
Conv2D::~Conv2D() {
    free(this->F);
    if (this->phase == TRAIN) {
        free(this->delta_buf);
    }
}

int Conv2D::configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase) {
    Layer::configure(batch, learning_rate, v_param, prevLayer, phase);
    // for filter and data matmul
    //this->X = (float*)malloc(sizeof(float)*this->batch*this->channel*this->kernel_size*this->kernel_size*this->units*this->units);
    this->Y = (float*)malloc(sizeof(float)*this->batch*this->filter*this->units);
    this->E = (float*)malloc(sizeof(float)*this->batch*this->channel*this->input_shape);
    this->delta_buf = (float*)malloc(sizeof(float)*this->batch*this->filter*this->units);
    this->F = (float*)malloc(sizeof(float)*this->filter*this->kernel_size*this->kernel_size);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> rand(-1.0,1.0);
    #pragma omp  parallel for
    for (int i = 0; i < this->filter*this->kernel_size*this->kernel_size; i++) {
        this->F[i] = rand(mt);
    }

    return 1;
}

void Conv2D::forward(float* x) {
    #pragma omp  parallel for
    for (int i = 0; i < this->batch*this->filter*this->units; i++) {
        this->Y[i] = 0;
    }
    #pragma omp  parallel for
    for (int b = 0; b < this->batch; b++) {
        for (int c = 0; c < this->channel; c++) {
            for (int f = 0; f < this->filter; f++) {
                for (int ro = 0; ro + this->kernel_size < this->i_rowcol; ro += this->stride) {
                    for (int co = 0; co + this->kernel_size < this->i_rowcol; co += this->stride) {
                        for (int ki = 0; ki < this->kernel_size; ki++) {
                            for (int kj = 0; kj < this->kernel_size; kj++) {
                                this->Y[b*this->units*this->filter+f*this->units+ro*this->i_rowcol+co] += \
                                    x[b*this->input_shape*this->channel+c*this->input_shape+ro*this->i_rowcol+co+ki*this->i_rowcol+kj] * \
                                    this->F[f*this->kernel_size*this->kernel_size+ki*this->kernel_size+kj];
                            }
                        }
                    }
                }
            }
        }
    }

    if (this->phase == TRAIN) {
        this->X = x;
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
                                this->E[b*this->channel*this->input_shape+c*this->input_shape + \
                                        ro*this->stride+co*this->stride+ki*this->i_rowcol+kj] += \
                                    this->Y[b*this->filter*this->units+f*this->units+ro*this->u_rowcol+co] * \
                                    this->F[f*this->kernel_size*this->kernel_size+ki*this->kernel_size+kj];
                            }
                        }
                    }
                }
            }
        }
    }

    // update
    if (this->momentum_a != 0) {
        #pragma omp  parallel for
        for (int b = 0; b < this->batch; b++) {
            for (int c = 0; c < this->channel; c++) {
                for (int f = 0; f < this->filter; f++) {
                    for (int ro = 0; ro < this->u_rowcol; ro++) {
                        for (int co = 0; co < this->u_rowcol; co++) {
                            for (int ki = 0; ki < this->kernel_size; ki++) {
                                for (int kj = 0; kj < this->kernel_size; kj++) {
                                    this->F[f*this->kernel_size*this->kernel_size+ki*this->kernel_size+kj] -= \
                                        this->momentum_a * this->delta_buf[b*this->filter*this->units+f*this->units+ro*this->u_rowcol+co] + \
                                        (this->learning_rate * this->X[b*this->channel*this->input_shape+c*this->input_shape+ \
                                                                       ro*this->i_rowcol+co+ki*this->i_rowcol+kj] * \
                                         e[b*this->filter*this->units+f*this->units+ro*this->u_rowcol+co])*this->batch_inv;
                                }
                            }
                        }
                    }
                }
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < this->batch*this->filter*this->units; i++) {
            this->delta_buf[i] = e[i];
        }
    } else {
        #pragma omp  parallel for
        for (int b = 0; b < this->batch; b++) {
            for (int c = 0; c < this->channel; c++) {
                for (int f = 0; f < this->filter; f++) {
                    for (int ro = 0; ro < this->u_rowcol; ro++) {
                        for (int co = 0; co < this->u_rowcol; co++) {
                            for (int ki = 0; ki < this->kernel_size; ki++) {
                                for (int kj = 0; kj < this->kernel_size; kj++) {
                                    this->F[f*this->kernel_size*this->kernel_size+ki*this->kernel_size+kj] -= \
                                        this->learning_rate * this->X[b*this->channel*this->input_shape+c*this->input_shape+ \
                                                                       ro*this->i_rowcol+co+ki*this->i_rowcol+kj] * \
                                         e[b*this->filter*this->units+f*this->units+ro*this->u_rowcol+co]*this->batch_inv;
                                }
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
    if (stride <= 0) {
        // warning
        this->stride = 1;
    }
    this->u_rowcol = (this->i_rowcol - kernel_size)/stride + 1;
    this->units = this->u_rowcol*this->u_rowcol;
    this->channel = channel;
}
MaxPooling2D::~MaxPooling2D() {
    if (this->phase == TRAIN) {
        free(this->L);
    }
}

int MaxPooling2D::configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase) {
    Layer::configure(batch, learning_rate, v_param, prevLayer, phase);
    this->Y = (float*)malloc(sizeof(float)*this->batch*this->channel*this->units);
    if (this->phase == TRAIN) {
        this->L = (int*)malloc(sizeof(int)*this->batch*this->channel*this->units);
        this->E = (float*)malloc(sizeof(float)*this->batch*this->channel*this->input_shape);
    }
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
                                if (this->phase == TRAIN) {
                                    this->L[b*this->channel*this->units+c*this->units+ro*this->u_rowcol+co] = ki*this->i_rowcol+kj;
                                }
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
                    this->E[b*this->input_shape*this->channel+c*this->input_shape+ro*this->i_rowcol+co+ \
                            this->L[b*this->units*this->channel+c*this->units+ro*this->u_rowcol+co]] = \
                        e[b*this->units*this->channel+c*this->units+ro*this->u_rowcol+co];
                }
            }
        }
    }
    return;
}


LSTM::LSTM(int input_shape, int units) : Layer(input_shape, units) {}
~LSTM::LSTM() {
    for (int i = 0; i < 4; i++) {
        free(this->Wx[this->Idx[i]]);
        free(this->Wh[this->Idx[i]]);
    }
}

int LSTM::configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase) {
    int ret = Layer::configure(batch, learning_rate, v_param, prevLayer, phase);
    this->acts['I'] = new Sigmoid();
    this->acts['F'] = new Sigmoid();
    this->acts['O'] = new Sigmoid();
    this->acts['U'] = new Tanh();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> rand(-1.0,1.0);
    for (int i = 0; i < 4; i++) {
        this->Wx[this->Idx[i]] = (float*)malloc(sizeof(float*)*this->input_shape*this->units);
        this->Wx[this->Idx[i]] = (float*)malloc(sizeof(float*)*this->units*this->units);
        #pragma omp  parallel for
        for (int iu = 0; iu < this->input_shape*this->units; iu++) {
            this->Wx[this->Idx[i]][iu] = rand(mt);
        }
        #pragma omp  parallel for
        for (int uu = 0; uu < this->units*this->units; uu++) {
            this->Wh[this->Idx[i]][uu] = rand(mt);
        }
        *this->B[this->Idx[i]] = rand(mt);
        this->acts[this->Idx].configure(batch, learning_rate, v_param, prevLayer, phase);
    }

    for (int i = 0; i < 8; i++) {
        this->buff[this->bIdx[i]] = (float*)malloc(sizeof(float*)*this->batch*this->units);
    }

    return ret;
}

void LSTM::forward(float* x) {
    for (int g = 0; i < 4; i++) {
        for (int b = 0; b < this->batch; b++) {
            for (int i = 0; i < this->input_shape; i++) {
                for (int u = 0; u < this->units; u++) {
                    this->buff[this->Idx[g]][b*this->units + u] += x[b*this->input_shape + i] * this->Wx[this->Idx[g]][i*this->units + u] + this->buff['h'][b*this->units + u] * this->Wh[this->Idx[g]][u*this->units + u] + *this->B;
                }
            }
        }
    }

    if (this->phase = TRAIN) {
        this->X = x;
    }
    this->Y = this->buff["H"];
}
