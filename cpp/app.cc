#include "layer.h"
#include "activation.h"
#include "loss.h"
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
    Conv2D* conv1 = new Conv2D(1024, 3, 4, 3, 1, 0);
    MaxPooling2D* pool1 = new MaxPooling2D(900, 4, 2, 1);
    Sigmoid* act1 = new Sigmoid();
    FullyConnect* fc1 = new FullyConnect(841*4, 841);
    ReLU* act2 = new ReLU();
    FullyConnect* fc2 = new FullyConnect(841, 10);
    Softmax* act3 = new Softmax();
    MSE* loss = new MSE();

    conv1->configure(10, learning_rate, nullptr);
    pool1->configure(10, learning_rate,  conv1);
    act1->configure(10, learning_rate, pool1);
    fc1->configure(10, learning_rate, act1);
    act2->configure(10, learning_rate, fc1);
    fc2->configure(10, learning_rate, act2);
    act3->configure(10, learning_rate, fc2);
    loss->configure(10, act3);

    // run
    for (int e = 0; e < epoch; e++) {
        for (int i = 0; i < 40000; i += batch) {
            conv1->forward(&all_data_float[i*3072]);
            pool1->forward(conv1->Y);
            act1->forward(pool1->Y);
            fc1->forward(act1->Y);
            act2->forward(fc1->Y);
            fc2->forward(act2->Y);
            act3->forward(fc2->Y);
            if (i % 1000 == 0) {
                float err = loss->error(act3->Y, &one_hot_label[i*CLASS]);
                std::cout << "loop " << i << " " << err << std::endl;
                for (int j = 0; j < 2; j++) {
                    std::cout << conv1->F[j] << std::endl;
                }
                for (int j = 0; j < 2; j++) {
                    std::cout << fc1->W[j] << std::endl;
                }
                std::cout << *fc1->B << std::endl;
                for (int j = 0; j < 2; j++) {
                    std::cout << fc2->W[j] << std::endl;
                }
                std::cout << *fc2->B << std::endl;
            }
            loss->partial_derivative(act3->Y, &one_hot_label[i*CLASS]);
            //for (int k = 0; k < batch*10; k++) {
            //std::cout << k << ":sdfkj " << ":" << loss->D[k] << std::endl;
            //}
            act3->backward(loss->D);
            fc2->backward(act3->E);
            act2->backward(fc2->E);
            fc1->backward(act2->E);
            act1->backward(fc1->E);
            pool1->backward(act1->E);
            conv1->backward(pool1->E);
        }
    }
        
    return 1;
}
