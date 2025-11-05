#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <vector>
using namespace std;

using Matrix = vector<vector<float>>;
using Tensor4D = vector<vector<vector<vector<float>>>>;

Matrix random_matrix(int rows, int cols, float mean = 0.0f, float stddev = 0.01f);
Tensor4D random_tensor_4d(int F, int C, int H, int W, float mean = 0.0f, float stddev = 0.01f);

#endif
