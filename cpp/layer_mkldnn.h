#ifndef CPP_LAYER_MKLDNN_H_
#define CPP_LAYER_MKLDNN_H_

#include "mkldnn.hpp"

using namespace mkldnn;

class Layer {
public:
    int batch;
    int filter;
    int channel;
    int input_shape;
    int units;
    Layer* prevLayer;
    Layer* nxtLayer;
    float learning_rate;
    primitive prim;
    primitive prim_bw;
    memory dst_mem;
    memory dst_mem_diff;
    memory::dims src_dim;
    memory::dims dst_dim;
    Layer(int input_shape, int units);
    virtual ~Layer();
    virtual int configure(int batch, float learning_rate, Layer* prevLayer, mkldnn::engine backend) = 0;
};

class FullyConnect : public Layer {
public:
    memory::dims weight_dim;
    memory::dims bias_dim; // TBD
    inner_product_forward::primitive_desc pd;
    inner_product_backward_weights::primitive_desc pd_diff;
    memory weight_mem;
    memory bias_mem;
    FullyConnect(int input_shape, int units);
    ~FullyConnect();
    int configure(int batch, float learning_rate, Layer* prevLayer, mkldnn::engine backend);
};

class Conv2D : public Layer {
public:
    memory::dims filter_dim;
    memory::dims bias_dim; // TBD
    memory::dims stride_dim;
    memory::dims padding_dim;
    convolution_forward::primitive_desc pd;
    convolution_backward_weights::primitive_desc pd_diff;
    memory filter_mem;
    memory bias_mem;
    int i_rowcol;
    int u_rowcol;
    int kernel_size;
    int stride;
    int padding;
    Conv2D(int input_shape, int channel, int filter, int kernel_size, int stride, int padding);
    ~Conv2D();
    int configure(int batch, float learning_rate, Layer* prevLayer, mkldnn::engine backend);
};

class MaxPooling2D : public Layer {
public:
    memory::dims kernel_dim;
    memory::dims stride_dim;
    memory::dims padding_dim;
    pooling_forward::primitive_desc pd;
    pooling_backward::primitive_desc pd_diff;
    memory workspace_mem;
    int i_rowcol;
    int u_rowcol;
    int kernel_size;
    int stride;
    MaxPooling2D(int input_shape, int channel, int kernel_size, int stride);
    ~MaxPooling2D();
    int configure(int batch, float learning_rate, Layer* prevLayer, mkldnn::engine backend);
};

#endif // CPP_LAYER_MKLDNN_H_
