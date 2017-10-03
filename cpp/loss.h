#ifndef CPP_LOSS_H_
#define CPP_LOSS_H_

#include "layer.h"
#include <vector>

using namespace std;

class Loss {
public:
    vector<float> D;
    int batch;
    float batch_inv;
    Layer* prevLayer;
    Loss();
    ~Loss();
    virtual int configure(int batch, Layer* prevLayer);
    virtual float error(vector<float> *x, int* label) = 0;
    virtual void partial_derivative(vector<float> *x, int* label) = 0;
};

class MSE : public Loss {
public:
    MSE();
    ~MSE();
    int configure(int batch, Layer* prevLayer);
    float error(vector<float> *x, int* label);
    void partial_derivative(vector<float> *x, int* label);
};

#endif // CPP_LOSS_H_
