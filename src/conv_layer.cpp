#include "../include/conv_layer.hpp"
#include <vector>
#include <stdexcept>

using namespace std;

// Compute single channel convolution (no padding, stride = 1)
float conv2d(const vector<vector<float>>& input,
             const vector<vector<float>>& kernel,
             int i_start, int j_start) {

    float sum = 0.0f;
    int kernal_height = kernel.size();
    int kernel_width = kernel[0].size();

    for (int i = 0; i < kernal_height; ++i)
        for (int j = 0; j < kernel_width; ++j)
            sum += input[i_start + i][j_start + j] * kernel[i][j];

    return sum;
}

// Forward pass for convolution layer
vector<vector<vector<float>>> conv_forward(
    const Tensor3D& input, // num_channels x img_height x img_width
    const Tensor4D& filters, // num_filters x num_channels x filter_height x filter_width
    const vector<float>& biases // num_filters
) {
    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();

    int F = filters.size();
    int filter_height = filters[0][0].size();
    int filter_width = filters[0][0][0].size();

    int outH = H - filter_height + 1;
    int outW = W - filter_width + 1;

    vector<vector<vector<float>>> output(F,
        vector<vector<float>>(outH, vector<float>(outW, 0.0f)));

    for (int f = 0; f < F; ++f) {
        for (int i = 0; i < outH; ++i) {
            for (int j = 0; j < outW; ++j) {
                float sum = 0.0f;
                for (int c = 0; c < C; ++c) {
                    sum += conv2d(input[c], filters[f][c], i, j);
                }
                output[f][i][j] = sum + biases[f];
            }
        }
    }

    return output;
}
