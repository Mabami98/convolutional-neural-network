#ifndef DATA_PREPROCESSING_HPP
#define DATA_PREPROCESSING_HPP

#include <vector>
#include <string>

using namespace std;

pair<vector<vector<float>>, vector<int>> load_dataset(const string& split);
vector<vector<float>> one_hot_encode(const vector<int>& y, int num_classes);
vector<float> compute_mean(const vector<vector<float>>& X);
vector<float> compute_std_dev(const vector<vector<float>>& X, const vector<float>& mean);
void normalize(vector<vector<float>>& X, const vector<float>& mean, const vector<float>& std);

// Converts [features][samples] X to [height, width, channels]
vector<vector<vector<vector<float>>>> convert_to_images(
    const vector<vector<float>>& X, int channels, int height, int width
);





#endif
