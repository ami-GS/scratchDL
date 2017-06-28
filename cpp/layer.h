#ifndef CPP_LAYER_H_
#define CPP_LAYER_H_

class Layer {
public:
    Layer(int input_shape, int units);
    virtual ~Layer();
    virtual configure();
    virtual forward();
    virtual backward();
};

class FullyConnect : public Layer {
public:
    FullyConnect(int input_shape, int units);
    ~FullyConnect();
};

#endif // CPP_LAYER_H_
