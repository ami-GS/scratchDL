#include "layer.h"
#include "activation.h"
#include "loss.h"
#include "network.h"
#include <iostream>
#include <fstream>
#define CLASS   10
#define NUMIMG  10000
#define IMGSZ   3072
#define NUMDSET 5

int all_label[NUMIMG];
int one_hot_label[NUMDSET*NUMIMG*CLASS];
float all_data_float[NUMDSET*IMGSZ*NUMIMG];
float learning_rate = 0.001;
int batch = 20;
int epoch = 2;

int main() {
    // read data
    char tmp_data[IMGSZ];
    char tmp_label;
    std::string f_prefix("../data/cifar10/data_batch_");
    std::string f_suffix(".bin");
    for (int b = 1; b < 6; b++) {
        std::ifstream infile(f_prefix + std::to_string(b) + f_suffix);
        for (int i = 0; i < NUMIMG; i++) {
            infile.read(&tmp_label, 1);
            all_label[i] = (int)tmp_label;
            for (int j = 0; j < 10; j++) {
                one_hot_label[b*NUMIMG + i*CLASS + j] = 0;
            }
            one_hot_label[b*NUMIMG+ i*CLASS + (int)tmp_label] = 1;
            infile.read(tmp_data, IMGSZ);
            for (int j = 0; j < IMGSZ; j++) {
                all_data_float[b*NUMIMG + i*IMGSZ+j] = ((float)(unsigned char)tmp_data[j])/255;
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
        for (int i = 0; i < NUMDSET*NUMIMG; i += batch) {
            float* data = &all_data_float[i*IMGSZ];
            int* label = &one_hot_label[i*CLASS];
            network->train(data, label);
            if (i % 800 == 0) {
                float err = loss->error(network->layers[6]->Y, label);
                std::cout << "loop " << i << " " << err << std::endl;
            }
        }
    }
    return 1;
}
