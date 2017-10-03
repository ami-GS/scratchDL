#include "layer.h"
#include "activation.h"
#include "loss.h"
#include "network.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

#define CLASS   10
#define NUMIMG  10000
#define IMGSZ   3072
#define NUMDSET 5
#define batch 20

int all_label[NUMIMG];
int one_hot_label[NUMDSET*NUMIMG*CLASS];
float all_data_float[NUMDSET*IMGSZ*NUMIMG];
// randomized idx
int idx_array[NUMIMG*NUMDSET];
// to randome data
int batch_label[batch*CLASS];
vector<float> batch_data_float(batch*IMGSZ);
float learning_rate = 0.001;
float momentum_param = 0.8;
int epoch = 2;

int main() {
    // read data
    char tmp_data[IMGSZ];
    char tmp_label;
    std::string f_prefix("../data/cifar10/data_batch_");
    std::string f_suffix(".bin");
    for (int b = 0; b < 5; b++) {
        std::ifstream infile(f_prefix + std::to_string(b+1) + f_suffix);
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

    for (int i = 0; i < NUMIMG*NUMDSET; i++)
        idx_array[i] = i;
    std::shuffle(idx_array, idx_array+NUMIMG*NUMDSET, std::mt19937());

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
    network->configure(batch, learning_rate, momentum_param);
    // run
    for (int e = 0; e < epoch; e++) {
        for (int i = 0; i < NUMDSET*NUMIMG; i += batch) {
            for (int b = 0; b < batch; b++) {
                for (int j = 0; j < IMGSZ; j++)
                    batch_data_float[b*IMGSZ+j] = all_data_float[idx_array[i+b]*IMGSZ+j];
                for (int j = 0; j < CLASS; j++)
                    batch_label[b*CLASS+j] = one_hot_label[idx_array[i+b]*CLASS+j];
            }

            vector<float>* data = &batch_data_float;
            int* label = batch_label;
            if (i % 800 == 0) {
                float err = loss->error(&network->layers[6]->Y, label);
                std::cout << "loop " << i << " " << err << std::endl;
            }
            network->train(data, label);
        }
    }
    return 1;
}
