#ifndef CPP_ACTIVATION_H_
#define CPP_ACTIVATION_H_

#include "layer.h"
#include <stdlib.h>

class Activation : public Layer {
public:
    Activation();
    ~Activation();
    int configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase);
    virtual void forward(vector<float> *x) = 0;
    virtual void backward(vector<float> *e) = 0;
};

class Sigmoid : public Activation {
public:
    Sigmoid();
    ~Sigmoid();
    void forward(vector<float> *x);
    void backward(vector<float> *e);
};

class ReLU : public Activation {
public:
    ReLU();
    ~ReLU();
    void forward(vector<float> *x);
    void backward(vector<float> *e);
};

class Softmax : public Activation {
public:
    float* maxVal;
    Softmax();
    ~Softmax();
    void forward(vector<float> *x);
    void backward(vector<float> *e);
};

#endif // CPP_ACTIVATION_H_
