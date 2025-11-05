#pragma once
#include <vector>
using namespace std;

// 3D tensor: channels × height × width
using Tensor3D = vector<vector<vector<float>>>;

// 4D tensor: num_filters × channels × height × width
using Tensor4D = vector<Tensor3D>;

using Matrix = vector<vector<float>>;                   

// Perform convolution (valid mode: no padding, stride=1)
vector<vector<vector<float>>> conv_forward(
    const Tensor3D& input,  // shape: C x H x W
    const Tensor4D& filters,  // shape: F x C x fH x fW
    const vector<float>& biases // shape: F
);
