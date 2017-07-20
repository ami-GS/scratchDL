#include "network_mkldnn.h"
#include "mkldnn.hpp"

using namespace mkldnn;

Network::Network(int layerNum, Layer** layers, Loss* loss, engine backend) : backend(backend) {
    for (int i = 0; i < layerNum; i++) {
        this->vec_fwd.push_back(layers[i]->prim);
    }
}


Network::~Network() {}

int Network::train() {
    stream(stream::kind::eager).submit(this->vec_fwd).wait();
    stream(stream::kind::eager).submit(this->vec_bwd).wait();

    return 1;
}

