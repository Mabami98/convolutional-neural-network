#include "../include/cnn_trainer.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>

using namespace std;

CNNTrainer::CNNTrainer(CNN& m,
                       const vector<vector<vector<vector<float>>>>& imgs,
                       const vector<vector<float>>& lbls,
                       float lr, int batch_sz, int ep)
    : model(m), images(imgs), labels(lbls),
      learning_rate(lr), batch_size(batch_sz), epochs(ep) {}

void CNNTrainer::train() {
    cout << "\nStarting CNN training...\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        int n = images.size();
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        shuffle(idx.begin(), idx.end(), mt19937(random_device{}()));

        float epoch_loss = 0.0f;
        int correct = 0;

        for (int i = 0; i < n; ++i) {
            const auto& img = images[idx[i]];
            const auto& label = labels[idx[i]];

            auto probs = model.forward(img);

            int true_id = distance(label.begin(), max_element(label.begin(), label.end())); 

            vector<vector<float>> lbl_mat(model.num_classes, vector<float>(1, 0.0f));
            lbl_mat[true_id][0] = 1.0f;

            epoch_loss += model.compute_loss(probs, lbl_mat);
            model.backward(probs, lbl_mat);
            model.update_weights(learning_rate);

            int pred = model.predict(img);
            if (pred == true_id) {
                correct++;
            }
        }

        cout << "Epoch " << epoch + 1 << "/" << epochs
             << " | Loss: " << epoch_loss / n
             << " | Accuracy: " << (float)correct / n * 100.0f << "%\n";
    }
}
