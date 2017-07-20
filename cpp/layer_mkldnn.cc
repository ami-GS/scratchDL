#include "layer_mkldnn.h"
#include "mkldnn.hpp"
#include <stdlib.h>
#include <random>
#include <iostream>

using namespace mkldnn;

Layer::Layer(int input_shape, int units) : batch(1), filter(0), channel(0), input_shape(input_shape), units(units), prevLayer(nullptr), dst_mem(primitive()), dst_mem_diff(primitive()) {}
Layer::~Layer() {}

FullyConnect::FullyConnect(int input_shape, int units) : Layer(input_shape, units), weight_mem(primitive()), bias_mem(primitive()), pd(inner_product_forward::primitive_desc()), pd_diff(inner_product_backward_weights::primitive_desc()) {}
FullyConnect::~FullyConnect() {}

int FullyConnect::configure(int batch, float learning_rate, Layer* prevLayer, engine backend) {
    this->batch = batch;
    this->learning_rate = learning_rate;
    if (prevLayer != nullptr) {
        prevLayer->nxtLayer = this;
        this->prevLayer = prevLayer;
    }

    this->src_dim = {batch, this->input_shape};
    this->dst_dim = {batch, this->units};
    this->weight_dim = {this->input_shape, this->units};

    auto src_md = memory::desc({this->src_dim}, memory::data_type::f32, memory::format::any);
    auto dst_md = memory::desc({this->dst_dim}, memory::data_type::f32, memory::format::any);
    auto weight_md = memory::desc({this->weight_dim}, memory::data_type::f32, memory::format::any);

    auto desc = inner_product_forward::desc(prop_kind::forward, src_md, weight_md, dst_md);
    this->pd = inner_product_forward::primitive_desc(desc, backend);

    this->dst_mem = memory::memory(this->pd.dst_primitive_desc());
    this->weight_mem = memory::memory(this->pd.weights_primitive_desc());

    this->prim = inner_product_forward(this->pd, this->prevLayer->dst_mem,
                                       this->weight_mem, this->dst_mem);

    //---------- for backward primitive -----------
    auto bwd_src_md = memory::desc({this->src_dim}, memory::data_type::f32, memory::format::any);
    auto diff_weights_md = memory::desc({this->weight_dim}, memory::data_type::f32, memory::format::any);
    auto diff_dst_md = memory::desc({this->dst_dim}, memory::data_type::f32, memory::format::any);

    auto fc_bwd_weights_desc = inner_product_backward_weights::desc(bwd_src_md, diff_weights_md, diff_dst_md);
    this->pd_diff = inner_product_backward_weights::primitive_desc(fc_bwd_weights_desc, backend, this->pd);

    this->dst_mem_diff = memory::memory(this->pd_diff.diff_weights_primitive_desc());

    this->prim_bw = inner_product_backward_weights(this->pd_diff, this->prevLayer->dst_mem, this->dst_mem, this->dst_mem_diff);

    return 1;
}


Conv2D::Conv2D(int input_shape, int channel, int filter, int kernel_size, int stride, int padding) : Layer(input_shape, 0), i_rowcol((int)std::sqrt((float)input_shape)), kernel_size(kernel_size), stride(stride), padding(padding) {
    if (stride <= 0) {
        // warning
        this->stride = 1;
    }
    this->u_rowcol = (this->i_rowcol + 2*padding - kernel_size)/stride + 1;
    this->units = this->u_rowcol*this->u_rowcol;
    this->channel = channel;
    this->filter = filter;
}
Conv2D::~Conv2D() {}

int Conv2D::configure(int batch, float learning_rate, Layer* prevLayer, engine backend) {
    this->batch = batch;
    this->learning_rate = learning_rate;
    if (prevLayer != nullptr) {
        prevLayer->nxtLayer = this;
        this->prevLayer = prevLayer;
    }

    this->src_dim = {batch, this->input_shape};
    this->dst_dim = {batch, this->filter, this->u_rowcol, this->u_rowcol};
    this->filter_dim = {this->filter, this->kernel_size, this->kernel_size};
    this->stride_dim = {this->stride, this->stride};
    this->padding_dim = {this->padding, this->padding};

    auto src_md = memory::desc({this->src_dim}, memory::data_type::f32, memory::format::any);
    auto dst_md = memory::desc({this->dst_dim}, memory::data_type::f32, memory::format::any);
    auto weights_md = memory::desc({this->filter_dim}, memory::data_type::f32, memory::format::any);

    auto desc = convolution_forward::desc(prop_kind::forward,
                                               convolution_direct,
                                               src_md, weights_md,
                                               dst_md, this->stride_dim,
                                               this->padding_dim, this->padding_dim,
                                               padding_kind::zero);
    this->pd = convolution_forward::primitive_desc(desc, backend);

    //auto conv_src_memory = memory::memory(this->pd.src_primitive_desc());
    this->dst_mem = memory::memory(this->pd.dst_primitive_desc());
    this->filter_mem = memory::memory(this->pd.weights_primitive_desc());

    this->prim = convolution_forward(this->pd, this->prevLayer->dst_mem, this->filter_mem, this->dst_mem);

    //---------- for backward primitive -----------
    auto bwd_src_md = memory::desc({this->src_dim}, memory::data_type::f32, memory::format::any);
    auto diff_dst_md = memory::desc({this->dst_dim}, memory::data_type::f32, memory::format::any);
    auto diff_weights_md = memory::desc({this->filter_dim}, memory::data_type::f32, memory::format::any);

    auto bwd_weights_desc = convolution_backward_weights::desc(
        convolution_direct, bwd_src_md, diff_weights_md,
        diff_dst_md, this->stride_dim, this->padding_dim, this->padding_dim, padding_kind::zero);
    this->pd_diff = convolution_backward_weights::primitive_desc(bwd_weights_desc, backend, this->pd);

    this->dst_mem_diff = memory::memory(this->pd_diff.diff_weights_primitive_desc());

    this->prim_bw = convolution_backward_weights(this->pd_diff, this->prevLayer->dst_mem, this->dst_mem, this->dst_mem_diff);

    return 1;
}

MaxPooling2D::MaxPooling2D(int input_shape, int channel, int kernel_size, int stride) : Layer(input_shape, 0), i_rowcol((int)std::sqrt((float)input_shape)), kernel_size(kernel_size), stride(stride) {
    if (stride <= 0) {
        // warning
        this->stride = 1;
    }
    this->u_rowcol = (this->i_rowcol - kernel_size)/stride + 1;
    this->units = this->u_rowcol*this->u_rowcol;
    this->channel = channel;
}
MaxPooling2D::~MaxPooling2D() {}

int MaxPooling2D::configure(int batch, float learning_rate, Layer* prevLayer, engine backend) {
    this->batch = batch;
    this->learning_rate = learning_rate;
    if (prevLayer != nullptr) {
        prevLayer->nxtLayer = this;
        this->prevLayer = prevLayer;
    }

    this->dst_dim = {batch, this->filter, this->u_rowcol, this->u_rowcol};
    this->kernel_dim = {this->kernel_size, this->kernel_size};
    this->stride_dim = {this->stride, this->stride};
    this->padding_dim = {0, 0}; // need padding

    auto pool_dst_md = memory::desc({this->dst_dim}, memory::data_type::f32, memory::format::any);
    auto pool_desc = pooling_forward::desc(prop_kind::forward,
                                                   pooling_max,
                                                   this->prevLayer->dst_mem.get_primitive_desc().desc(),
                                                   pool_dst_md, this->stride_dim, this->kernel_dim,
                                                   this->padding_dim, this->padding_dim,
                                                   padding_kind::zero);
    this->pd = pooling_forward::primitive_desc(pool_desc, backend);

    this->dst_mem = memory::memory(this->pd.dst_primitive_desc());
    this->workspace_mem = memory::memory(this->pd.workspace_primitive_desc());

    this->prim = pooling_forward(this->pd, this->prevLayer->dst_mem, this->dst_mem, this->workspace_mem);

    //---------- for backward primitive -----------
    auto pool_bwd_md = this->nxtLayer->dst_mem.get_primitive_desc().desc();
    auto pool_diff_dst_md = memory::desc({this->dst_dim}, memory::data_type::f32, memory::format::any);

    auto pool_bwd_desc = pooling_backward::desc(
        pooling_max, pool_bwd_md, pool_diff_dst_md, this->stride_dim,
        this->kernel_dim, this->padding_dim, this->padding_dim, padding_kind::zero);
    this->pd_diff = pooling_backward::primitive_desc(pool_bwd_desc, backend, this->pd);

    // TODO : the NULL should be replaced to data
    this->dst_mem_diff = memory::memory({{{this->dst_dim}, memory::f32, memory::format::nchw}, backend}, NULL);

    this->prim_bw = pooling_backward(this->pd_diff, this->dst_mem_diff, this->workspace_mem, this->prevLayer->dst_mem);

    return 1;
}

