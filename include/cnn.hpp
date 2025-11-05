#ifndef CNN_HPP
#define CNN_HPP

#include <vector>
using namespace std;

using Tensor3D = vector<vector<vector<float>>>;
using Tensor4D = vector<Tensor3D>;
using Matrix = vector<vector<float>>;

class CNN {
public:
    CNN(int in_channels, int in_height, int in_width,
        int filter_sz, int num_filt, int num_cls);

    Matrix forward(const Tensor3D& input);
    int predict(const Tensor3D& input);
    float compute_loss(const Matrix& probs, const Matrix& labels);
    void backward(const Matrix& probs, const Matrix& labels);
    void update_weights(float lr);

    int num_classes;

private:
    int input_channels, input_height, input_width;
    int filter_size, num_filters;

    Tensor4D filters;
    Matrix W;
    std::vector<float> b;

    Tensor3D last_input;
    Tensor3D last_conv_out;
    std::vector<float> last_flat;

    Tensor4D d_filters;
    Matrix dW;
    std::vector<float> db;
};

#endif
