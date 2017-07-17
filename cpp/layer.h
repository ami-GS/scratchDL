#ifndef CPP_LAYER_H_
#define CPP_LAYER_H_

class Layer {
public:
    int batch;
    int filter;
    int channel;
    int input_shape;
    int units;
    Layer* prevLayer;
    float learning_rate;
    float* E;
    float* Y;
    float* X;
    Layer(int input_shape, int units);
    virtual ~Layer();
    virtual int configure(int batch, float learning_rate, Layer* prevLayer) = 0;
    virtual int configure_mkldnn(int batch, float learning_rate, Layer* prevLayer) = 0;
    virtual void forward(float* x) = 0;
    virtual void backward(float* e) = 0;
};

class FullyConnect : public Layer {
public:
    float* W;
    float* B;
    FullyConnect(int input_shape, int units);
    ~FullyConnect();
    int configure(int batch, float learning_rate, Layer* prevLayer);
    int configure_mkldnn(int batch, float learning_rate, Layer* prevLayer);
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
    int configure(int batch, float learning_rate, Layer* prevLayer);
    int configure_mkldnn(int batch, float learning_rate, Layer* prevLayer);
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
    int configure(int batch, float learning_rate, Layer* prevLayer);
    int configure_mkldnn(int batch, float learning_rate, Layer* prevLayer);
    void forward(float* x);
    void backward(float* e);
};

#endif // CPP_LAYER_H_
