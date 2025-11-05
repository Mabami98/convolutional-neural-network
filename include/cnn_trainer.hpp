#ifndef CNN_TRAINER_HPP
#define CNN_TRAINER_HPP

#include "cnn.hpp"
#include <vector>

using namespace std;

class CNNTrainer {
public:
    CNNTrainer(CNN& model,
               const vector<Tensor3D>& images,
               const vector<vector<float>>& labels,
               float learning_rate, int batch_size, int epochs);

    void train();

private:
    CNN& model;
    const vector<Tensor3D>& images;
    const vector<vector<float>>& labels;
    float learning_rate;
    int batch_size;
    int epochs;
};

#endif
