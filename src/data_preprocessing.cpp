#include "../include/data_preprocessing.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cmath>
#include <vector>
#include <string>

using namespace std;

// read shape info from .shape file (e.g. 3072,10000)
pair<int, int> read_shape(const string& shape_file) {
    ifstream file(shape_file);
    if (!file) {
        cerr << "Error: Cannot open shape file: " << shape_file << endl;
        exit(1);
    }

    string line;
    getline(file, line);
    stringstream ss(line);
    string token;
    vector<int> dims;

    while (getline(ss, token, ',')) {
        dims.push_back(stoi(token));
    }

    if (dims.size() == 1) return make_pair(dims[0], 1);
    if (dims.size() == 2) return make_pair(dims[0], dims[1]);

    cerr << "Error: Invalid shape in file: " << shape_file << endl;
    exit(1);
}

// loads a binary float matrix (shape is in .shape file)
vector<vector<float>> load_matrix(const string& base_name) {
    auto [rows, cols] = read_shape(base_name + ".shape");

    if (rows == 0 || cols == 0) {
        cerr << "Error: Invalid matrix shape for " << base_name << ": " << rows << " x " << cols << endl;
        exit(1);
    }

    ifstream file(base_name + ".bin", ios::binary);
    vector<float> flat(rows * cols);
    file.read(reinterpret_cast<char*>(flat.data()), flat.size() * sizeof(float));

    vector<vector<float>> matrix(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = flat[i * cols + j];

    return matrix;
}

// loads a 1d int vector from file
vector<int> load_vector(const string& base_name) {
    auto [length, _] = read_shape(base_name + ".shape");
    ifstream file(base_name + ".bin", ios::binary);
    vector<int> vec(length);
    file.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(int));
    return vec;
}

// loads the actual dataset (X and y) based on split
pair<vector<vector<float>>, vector<int>> load_dataset(const string& split) {
    string base_path = "/Users/martin/Desktop/conv_net/datasets/";
    string folder = (split == "train") ? "train_data" : "test_data";
    string prefix = base_path + folder + "/";

    auto X = load_matrix(prefix + (split == "train" ? "X_train" : "X_test"));
    auto y = load_vector(prefix + (split == "train" ? "y_train" : "y_test"));

    cout << "Loaded " << split << " set: (" << X[0].size() << " images)" << endl;
    cout << "  X shape: " << X.size() << " x " << X[0].size() << endl;
    cout << "  y size:  " << y.size() << endl;

    return {X, y};
}

// convert labels to one-hot format
vector<vector<float>> one_hot_encode(const vector<int>& y, int num_classes) {
    int num_samples = y.size();
    vector<vector<float>> Y(num_samples, vector<float>(num_classes, 0.0f));

    for (int i = 0; i < num_samples; ++i) {
        int label = y[i];
        Y[i][label] = 1.0f;
    }

    return Y;
}

// compute mean pixel value per image
vector<float> compute_mean(const vector<vector<float>>& X) {
    int rows = X.size();
    int cols = X[0].size();
    vector<float> mean(rows, 0.0f);

    for (int i = 0; i < rows; ++i)
        mean[i] = accumulate(X[i].begin(), X[i].end(), 0.0f) / cols;

    return mean;
}

// compute std deviation per image (needed for normalization)
vector<float> compute_std_dev(const vector<vector<float>>& X, const vector<float>& mean) {
    int rows = X.size();
    int cols = X[0].size();
    vector<float> std(rows, 0.0f);

    for (int i = 0; i < rows; ++i) {
        float sum_sq = 0.0f;
        for (float val : X[i]) {
            float diff = val - mean[i];
            sum_sq += diff * diff;
        }
        std[i] = sqrt(sum_sq / cols);
        if (std[i] < 1e-6f) std[i] = 1.0f; // avoid division by zero
    }

    return std;
}

// normalize the matrix (zero mean, unit std)
void normalize(vector<vector<float>>& X, const vector<float>& mean, const vector<float>& std) {
    int rows = X.size();
    int cols = X[0].size();

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            X[i][j] = (X[i][j] - mean[i]) / std[i];
}

// reshape flat matrix X into 4d images
vector<vector<vector<vector<float>>>> convert_to_images(
    const vector<vector<float>>& X, int channels, int height, int width
) {
    int num_samples = X[0].size();
    vector<vector<vector<vector<float>>>> images(
        num_samples,
        vector<vector<vector<float>>>(
            channels, vector<vector<float>>(
                height, vector<float>(width)))
    );

    for (int i = 0; i < num_samples; ++i) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int flat_index = c * height * width + h * width + w;
                    images[i][c][h][w] = X[flat_index][i];
                }
            }
        }
    }

    return images;
}
