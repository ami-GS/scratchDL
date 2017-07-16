#include "layer.h"
#include "activation.h"
#include "loss.h"
#include "network.h"
#include <iostream>
#include <fstream>
#define CLASS 10

int all_label[10000];
int one_hot_label[4*10000*CLASS];
float all_data_float[4*3072*10000];
float learning_rate = 0.001;
int batch = 10;
int epoch = 2;

int main() {
    // read data
    char tmp_data[3072];
    char tmp_label;
    std::string f_prefix("../data/cifar10/data_batch_");
    std::string f_suffix(".bin");
    for (int b = 0; b < 4; b++) {
        std::ifstream infile(f_prefix + std::to_string(b) + f_suffix);
        for (int i = 0; i < 10000; i++) {
            infile.read(&tmp_label, 1);
            all_label[i] = (int)tmp_label;
            for (int j = 0; j < 10; j++) {
                one_hot_label[b*10000 + i*CLASS + j] = 0;
            }
            one_hot_label[b*10000+ i*CLASS + (int)tmp_label] = 1;
            infile.read(tmp_data, 3072);
            for (int j = 0; j < 3072; j++) {
                all_data_float[b*10000 + i*3072+j] = ((float)(unsigned char)tmp_data[j])/255;
            }
        }
    }

    // configure layer
    Layer* layers[7] = {
        new Conv2D(1024, 3, 4, 3, 1, 0),
        new MaxPooling2D(900, 4, 2, 1),
        new Sigmoid(),
        new FullyConnect(841*4, 841),
        new ReLU(),
        new FullyConnect(841, 10),
        new Softmax()
    };
    MSE* loss = new MSE();
    Network* network = new Network(7, layers, loss);
    network->configure(batch, learning_rate);

    // run
    for (int e = 0; e < epoch; e++) {
        for (int i = 0; i < 40000; i += batch) {
            float* data = &all_data_float[i*3072];
            int* label = &one_hot_label[i*CLASS];
            network->train(data, label);
            if (i % 800 == 0) {
                float err = loss->error(data, &one_hot_label[i*CLASS]);
                std::cout << "loop " << i << " " << err << std::endl;
            }
        }
    }
    return 1;
}
