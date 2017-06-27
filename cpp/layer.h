class Layer {
public:
    Layer(int input_shape, int unit);
    virtual ~Layer();
    virtual configure();
    virtual forward();
    virtual backward();
};

class FullyConnect : public Later {
public:
    FullyConnect(int units, int input_shape);
    ~FullyConnect();
}
