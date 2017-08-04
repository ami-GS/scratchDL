#ifndef CPP_ACTIVATION_H_
#define CPP_ACTIVATION_H_

#include "layer.h"
#include <stdlib.h>

class Activation : public Layer {
public:
    Activation();
    ~Activation();
    int configure(int batch, float learning_rate, float v_param, Layer* prevLayer);
    virtual void forward(float* x) = 0;
    virtual void backward(float* e) = 0;
};

class Sigmoid : public Activation {
public:
    Sigmoid();
    ~Sigmoid();
    void forward(float* x);
    void backward(float* e);
};

class ReLU : public Activation {
public:
    ReLU();
    ~ReLU();
    void forward(float* x);
    void backward(float* e);
};

class Softmax : public Activation {
public:
    Softmax();
    ~Softmax();
    void forward(float* x);
    void backward(float* e);
};

#endif // CPP_ACTIVATION_H_
