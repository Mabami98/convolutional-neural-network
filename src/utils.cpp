#include "../include/utils.hpp"

using namespace std;

// Flattens 3D tensor from [H, W, C] --> [F], where F = H x W x C
vector<float> flatten(const Tensor3D& tensor) {
    vector<float> flat;
    for (const auto& channel : tensor)
        for (const auto& row : channel)
            for (float val : row)
                flat.push_back(val);
    return flat;
}


vector<int> argmax(const vector<std::vector<float>>& scores) {
    int batch_size = scores[0].size();
    int num_classes = scores.size();
    vector<int> predictions(batch_size);
    float small_number = -1e9f;

    // Loop that finds the index 'i' of the highest score, in order to compute which class it belongs to
    for (int i = 0; i < batch_size; ++i) {
        float max_val = small_number;
        int best_class = -1;
        for (int c = 0; c < num_classes; ++c) {
            if (scores[c][i] > max_val) {
                max_val = scores[c][i];
                best_class = c;
            }
        }
        predictions[i] = best_class;
    }

    return predictions;
}

void print_tensor_shape(const Tensor3D& tensor) {
    cout << "Tensor shape: (" 
         << tensor.size() 
         << ", "
         << tensor[0].size() 
         << ", "
         << tensor[0][0].size() 
         << ")\n";
}