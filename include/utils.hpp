#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <iostream>
using namespace std;

using Tensor3D = vector<vector<vector<float>>>;

vector<float> flatten(const Tensor3D& tensor);
vector<int> argmax(const vector<vector<float>>& scores);
void print_tensor_shape(const Tensor3D& tensor);

#endif
