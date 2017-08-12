#ifndef CPP_LOSS_H_
#define CPP_LOSS_H_

#include "layer.h"


class Loss {
public:
    float* D;
    int batch;
    float batch_inv;
    Layer* prevLayer;
    Loss();
    ~Loss();
    virtual int configure(int batch, Layer* prevLayer);
    virtual float error(float* x, int* label) = 0;
    virtual void partial_derivative(float* x, int* label) = 0;
};

class MSE : public Loss {
public:
    MSE();
    ~MSE();
    int configure(int batch, Layer* prevLayer);
    float error(float* x, int* label);
    void partial_derivative(float* x, int* label);
};

#endif // CPP_LOSS_H_
