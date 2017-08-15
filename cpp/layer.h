#ifndef CPP_LAYER_H_
#define CPP_LAYER_H_

#include <map>

typedef enum {
    TRAIN,
    TEST,
} phase_t;

class Layer {
public:
    int batch;
    float batch_inv;
    int filter;
    int channel;
    int input_shape;
    int units;
    Layer* prevLayer;
    Layer* nxtLayer;
    float learning_rate;
    phase_t phase;
    float* E;
    // for Momentum
    float* delta_buf;
    float momentum_a;
    float* Y;
    float* X;
    Layer(int input_shape, int units);
    virtual ~Layer();
    virtual int configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase);
    virtual void forward(float* x) = 0;
    virtual void backward(float* e) = 0;
};

class FullyConnect : public Layer {
public:
    float* W;
    float* B;
    FullyConnect(int input_shape, int units);
    ~FullyConnect();
    int configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase);
    void forward(float* x);
    void backward(float* e);
};

class Conv2D : public Layer {
public:
    float* F;
    int i_rowcol;
    int u_rowcol;
    int kernel_size;
    int stride;
    int padding;
    Conv2D(int input_shape, int channel, int filter, int kernel_size, int stride, int padding);
    ~Conv2D();
    int configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase);
    void forward(float* x);
    void backward(float* e);
};

class MaxPooling2D : public Layer {
public:
    int* L; // locations
    int i_rowcol;
    int u_rowcol;
    int kernel_size;
    int stride;
    MaxPooling2D(int input_shape, int channel, int kernel_size, int stride);
    ~MaxPooling2D();
    int configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase);
    void forward(float* x);
    void backward(float* e);
};


class LSTM : public Layer {
public:
    char Idx[4] = {'I', 'F', 'U', 'O'};
    std::map<char, *Activation> acts;
    std::map<char, float*> Wx;
    std::map<char, float*> Wh;
    std::map<char, float> B;
    char bIdx[8] = {'C', 'c', 'H', 'h', 'I', 'F', 'U', 'O'};
    std::map<char, float*> buff;
    LSTM(int input_shape, int units);
    ~LSTM();
    int configure(int batch, float learning_rate, float v_param, Layer* prevLayer, phase_t phase);
    void forward(float* x);
    void backward(float* e);
};

#endif // CPP_LAYER_H_
