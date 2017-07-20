#ifndef CPP_ACTIVATION_MKLDNN_H_
#define CPP_ACTIVATION_MKLDNN_H_

#include "mkldnn.hpp"
#include "layer_mkldnn.h"
#include <stdlib.h>

class Activation : public Layer {
public:
    Activation();
    ~Activation();
    int configure(int batch, float learning_rate, Layer* prevLayer, mkldnn::engine backend);
};

class ReLU : public Activation {
public:
    ReLU();
    ~ReLU();
};

class Softmax : public Activation {
public:
    Softmax();
    ~Softmax();
};



#endif
