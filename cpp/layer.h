#ifndef CPP_LAYER_H_
#define CPP_LAYER_H_

class Layer {
public:
    int batch;
    int input_shape;
    int units;
    Layer* prevLayer;
    float* E;
    float* W;
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
    FullyConnect(int input_shape, int units);
    ~FullyConnect();
    int configure(int batch, Layer* prevLayer);
    void forward(float* x);
    void backward(float* e);
};

#endif // CPP_LAYER_H_
