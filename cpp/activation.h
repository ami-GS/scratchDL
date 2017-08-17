#ifndef CPP_ACTIVATION_H_
#define CPP_ACTIVATION_H_

#include "layer.h"
#include <stdlib.h>

class Activation : public Layer {
public:
    Activation();
    ~Activation();
    virtual int configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase);
    virtual void forward(float* x) = 0;
    virtual void backward(float* e) = 0;
};

class Sigmoid : public Activation {
public:
    Sigmoid();
    ~Sigmoid();
    int configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase);
    void forward(float* x);
    void backward(float* e);
};

class ReLU : public Activation {
public:
    ReLU();
    ~ReLU();
    int configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase);
    void forward(float* x);
    void backward(float* e);
};

class Softmax : public Activation {
public:
    float* maxVal;
    Softmax();
    ~Softmax();
    int configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase);
    void forward(float* x);
    void backward(float* e);
};

#endif // CPP_ACTIVATION_H_
