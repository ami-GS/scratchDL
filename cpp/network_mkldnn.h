#ifndef CPP_NETWORK_MKLDNN_H_
#define CPP_NETWORK_MKLDNN_H_

#include "layer_mkldnn.h"
#include "loss_mkldnn.h"
#include "mkldnn.hpp"

class Network {
public:
    std::vector<mkldnn::primitive> vec_fwd;
    std::vector<mkldnn::primitive> vec_bwd;
    mkldnn::engine backend;
    Network(int layerNum, Layer** layers, Loss* loss, mkldnn::engine backend);
    ~Network();
    int train();
};

#endif //CPP_NETWORK_MKLDNN_H_
