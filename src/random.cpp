#include "random.hpp"
#include <random>

using namespace std;


// Generates a random matrix, where each entry is drawn from the normal distribution
Matrix random_matrix(int num_rows, int num_cols, float mean, float stddev) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(mean, stddev);

    Matrix mat(num_rows, std::vector<float>(num_cols));
    for (int i = 0; i < num_rows; ++i)
        for (int j = 0; j < num_cols; ++j)
            mat[i][j] = dist(gen);

    return mat;
}


// Generates a random tensor, where each entry is drawn from the normal distribution
Tensor4D random_tensor_4d(int F, int C, int H, int W, float mean, float stddev) {
    random_device rd;
    mt19937 gen(rd()); // rng
    normal_distribution<float> dist(mean, stddev);

    Tensor4D tensor(F,
        vector<vector<vector<float>>>(C,
            vector<vector<float>>(H,
                vector<float>(W))));

    for (int f = 0; f < F; ++f)
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    tensor[f][c][i][j] = dist(gen);

    return tensor;
}