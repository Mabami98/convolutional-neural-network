#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <vector>

using namespace std;

void relu_inplace(vector<vector<vector<float>>>& tensor);

vector<vector<float>> softmax(const vector<vector<float>>& scores);

vector<vector<float>> softmax_cross_entropy_derivative(
    const vector<vector<float>>& probs,
    const vector<vector<float>>& labels);

#endif