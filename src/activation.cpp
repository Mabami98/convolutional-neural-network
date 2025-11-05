#include "activation.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

// relu activation applied in place
// relu(x) = max(0, x)
void relu_inplace(vector<vector<vector<float>>>& tensor) {
    for (auto& channel : tensor)
        for (auto& row : channel)
            for (auto& val : row)
                val = max(0.0f, val);
}

// softmax converts raw scores into probabilities
// softmax(xi) = exp(xi) / sumj exp(xj)
vector<vector<float>> softmax(const vector<vector<float>>& scores) {
    int num_classes = scores.size();
    int batch_size = scores[0].size();
    vector<vector<float>> probs(num_classes, vector<float>(batch_size));

    for (int i = 0; i < batch_size; ++i) {
        float max_val = -1e9f;

        // for numerical stability
        for (int c = 0; c < num_classes; ++c)
            max_val = max(max_val, scores[c][i]);

        float sum = 0.0f;

        for (int c = 0; c < num_classes; ++c) {
            probs[c][i] = exp(scores[c][i] - max_val);
            sum += probs[c][i];
        }

        for (int c = 0; c < num_classes; ++c)
            probs[c][i] /= sum;
    }

    return probs;
}

// derivative of cross entropy loss with softmax output
vector<vector<float>> softmax_cross_entropy_derivative(
    const vector<vector<float>>& probs,
    const vector<vector<float>>& labels) {

    int num_classes = probs.size();
    int batch_size = probs[0].size();
    vector<vector<float>> grad(num_classes, vector<float>(batch_size));

    for (int c = 0; c < num_classes; ++c)
        for (int i = 0; i < batch_size; ++i)
            grad[c][i] = probs[c][i] - labels[c][i];

    return grad;
}
