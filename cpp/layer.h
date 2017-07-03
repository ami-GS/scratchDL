#ifndef CPP_LAYER_H_
#define CPP_LAYER_H_

class Layer {
public:
    int batch;
    int input_shape;
    int units;
    Layer* prevLayer;
    float* E;
    float* Y;
    float* X;
    Layer(int input_shape, int units);
    virtual ~Layer();
    virtual int configure(int batch, Layer* prevLayer) = 0;
    virtual void forward(float* x) = 0;
    virtual void backward(float* e) = 0;
};

class FullyConnect : public Layer {
public:
    float* W;
    FullyConnect(int input_shape, int units);
    ~FullyConnect();
    int configure(int batch, Layer* prevLayer);
    void forward(float* x);
    void backward(float* e);
};

class Conv2D : public Layer {
public:
    float* F;
    int filter;
    int channel;
    int i_rowcol;
    int u_rowcol;
    int kernel_size;
    int stride;
    int padding;
    Conv2D(int input_shape, int channel, int filter, int kernel_size, int stride, int padding);
    ~Conv2D();
    int configure(int batch, Layer* prevLayer);
    void forward(float* x);
    void backward(float* e);
};

class MaxPooling2D : public Layer {
public:
    int* L; // locations
    int i_rowcol;
    int u_rowcol;
    int channel;
    int kernel_size;
    int stride;
    MaxPooling2D(int input_shape, int channel, int kernel_size, int stride);
    ~MaxPooling2D();
    int configure(int batch, Layer* prevLayer);
    void forward(float* x);
    void backward(float* e);
};

#endif // CPP_LAYER_H_
