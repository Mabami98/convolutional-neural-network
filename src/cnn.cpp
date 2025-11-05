#include "../include/cnn.hpp"
#include <random>
#include <iostream>

using namespace std;

// Utility functions
static Tensor4D random_tensor_4d(int F, int C, int H, int W, float mean=0.0f, float stddev=0.01f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);

    Tensor4D tensor(F, Tensor3D(C, std::vector<std::vector<float>>(H, std::vector<float>(W))));
    for (int f = 0; f < F; ++f)
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    tensor[f][c][i][j] = dist(gen);
    return tensor;
}

static Matrix random_matrix(int rows, int cols, float mean=0.0f, float stddev=0.01f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);

    Matrix mat(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat[i][j] = dist(gen);
    return mat;
}

static std::vector<float> flatten(const Tensor3D& tensor) {
    std::vector<float> flat;
    for (const auto& channel : tensor)
        for (const auto& row : channel)
            for (float val : row)
                flat.push_back(val);
    return flat;
}

static void relu_inplace(Tensor3D& tensor) {
    for (auto& channel : tensor)
        for (auto& row : channel)
            for (auto& val : row)
                val = std::max(0.0f, val);
}

static Matrix softmax(const Matrix& scores) {
    int num_classes = scores.size();
    int batch_size = scores[0].size();
    Matrix probs(num_classes, vector<float>(batch_size));

    for (int i = 0; i < batch_size; ++i) {
        float max_val = -1e9f;
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

static float conv2d_single(const vector<vector<float>>& input,
                           const vector<vector<float>>& kernel,
                           int i_start, int j_start) {
    float sum = 0.0f;
    int kh = kernel.size(), kw = kernel[0].size();
    for (int i = 0; i < kh; ++i)
        for (int j = 0; j < kw; ++j)
            sum += input[i_start + i][j_start + j] * kernel[i][j];
    return sum;
}

static Tensor3D conv_forward(const Tensor3D& input, const Tensor4D& filters, const vector<float>& biases) {
    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();
    int F = filters.size();
    int fH = filters[0][0].size();
    int fW = filters[0][0][0].size();
    int outH = H - fH + 1;
    int outW = W - fW + 1;

    Tensor3D output(F, vector<vector<float>>(outH, vector<float>(outW, 0.0f)));
    for (int f = 0; f < F; ++f)
        for (int i = 0; i < outH; ++i)
            for (int j = 0; j < outW; ++j) {
                float sum = 0.0f;
                for (int c = 0; c < C; ++c)
                    sum += conv2d_single(input[c], filters[f][c], i, j);
                output[f][i][j] = sum + biases[f];
            }
    return output;
}

CNN::CNN(int in_channels, int in_height, int in_width,
         int filter_sz, int num_filt, int num_cls)
    : input_channels(in_channels), input_height(in_height), input_width(in_width),
      filter_size(filter_sz), num_filters(num_filt), num_classes(num_cls) {

    filters = random_tensor_4d(num_filters, input_channels, filter_size, filter_size);
    int conv_output = input_height - filter_size + 1;
    int flat_size = num_filters * conv_output * conv_output;

    W = random_matrix(num_classes, flat_size); // Initialize weights randomly, sampled from a gaussian/normal distribution
    b = vector<float>(num_classes, 0.0f); // Initialize bias vector as 0s

    d_filters = Tensor4D(num_filters, Tensor3D(input_channels, vector<vector<float>>(filter_size, vector<float>(filter_size, 0.0f))));
    dW = Matrix(num_classes, vector<float>(flat_size, 0.0f));
    db = vector<float>(num_classes, 0.0f);
}

Matrix CNN::forward(const Tensor3D& input) {
    last_input = input;
    last_conv_out = conv_forward(input, filters, vector<float>(num_filters, 0.0f));
    relu_inplace(last_conv_out);
    last_flat = flatten(last_conv_out);

    Matrix scores(num_classes, vector<float>(1));
    for (int c = 0; c < num_classes; ++c) {
        float sum = b[c];
        for (size_t i = 0; i < last_flat.size(); ++i)
            sum += W[c][i] * last_flat[i];
        scores[c][0] = sum;
    }

    return softmax(scores);
}

int CNN::predict(const Tensor3D& input) {
    auto probs = forward(input); // Compute class probabilities for an input image (32, 32, 3)
    int best = 0;
    float maxp = probs[0][0];
    for (int c = 1; c < num_classes; ++c) { 
        if (probs[c][0] > maxp) {
            maxp = probs[c][0];
            best = c; // Take index of the highest probability, that is the class that the model thinks is most probable
        }
    }
    return best;
}

// Computes cross entropy loss 
float CNN::compute_loss(const Matrix& probs, const Matrix& labels) {
    float loss = 0.0f;
    float small_number = 1e-9f; // For numerical stability
    for (int c = 0; c < num_classes; ++c) {
        if (labels[c][0] == 1.0f)
            loss -= log(probs[c][0] + small_number);
    }
    return loss;
}

// Computes gradients based on predictions from the forward pass
void CNN::backward(const Matrix& probs, const Matrix& labels) {
    for (int c = 0; c < num_classes; ++c) {
        float grad = probs[c][0] - labels[c][0];
        db[c] += grad;
        for (size_t i = 0; i < last_flat.size(); ++i)
            dW[c][i] += grad * last_flat[i];
    }

    vector<float> dflat(last_flat.size(), 0.0f);
    for (int c = 0; c < num_classes; ++c)
        for (size_t i = 0; i < last_flat.size(); ++i)
            dflat[i] += (probs[c][0] - labels[c][0]) * W[c][i];

    int conv_out_h = last_conv_out[0].size();
    int conv_out_w = last_conv_out[0][0].size();


    // Update gradients of the filters/kernels
    Tensor3D dconv(num_filters, vector<vector<float>>(conv_out_h, vector<float>(conv_out_w, 0.0f)));
    int idx = 0;
    for (int f = 0; f < num_filters; ++f)
        for (int h = 0; h < conv_out_h; ++h)
            for (int w = 0; w < conv_out_w; ++w)
                dconv[f][h][w] = (last_conv_out[f][h][w] > 0.0f) ? dflat[idx++] : 0.0f;

    for (int f = 0; f < num_filters; ++f)
        for (int c = 0; c < input_channels; ++c)
            for (int i = 0; i < filter_size; ++i)
                for (int j = 0; j < filter_size; ++j)
                    d_filters[f][c][i][j] += dconv[f][i][j] * last_input[c][i][j];
}

void CNN::update_weights(float lr) {
    for (int c = 0; c < num_classes; ++c) {
        b[c] -= lr * db[c];
        db[c] = 0.0f; // gradient dJ/db
        for (size_t i = 0; i < W[c].size(); ++i) {
            W[c][i] -= lr * dW[c][i];
            dW[c][i] = 0.0f; // gradient dJ/dW
        }
    }

    for (int f = 0; f < num_filters; ++f)
        for (int c = 0; c < input_channels; ++c)
            for (int i = 0; i < filter_size; ++i)
                for (int j = 0; j < filter_size; ++j) {
                    filters[f][c][i][j] -= lr * d_filters[f][c][i][j];
                    d_filters[f][c][i][j] = 0.0f;
                }
}