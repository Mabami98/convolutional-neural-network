#include "../include/cnn.hpp"
#include "../include/cnn_trainer.hpp"
#include "../include/data_preprocessing.hpp"
#include "../include/performance_summary.hpp"
#include "../include/data_preprocessing.hpp"

#include <iostream>
#include <vector>

using namespace std;

int main() {
    // Fixed parameters
    const int num_classes    = 10;
    const int image_channels = 3;
    const int image_height   = 32;
    const int image_width    = 32;
    

    // Hyperparameters for tuning
    const int filter_size = 3;
    const int num_filters = 8;
    const float learning_rate = 0.01f;
    const int batch_size = 1; 
    const int epochs = 10;

    cout << "Loading data...\n";
    auto [X_train, y_train] = load_dataset("train");

    cout << "Normalizing data...\n";
    /* mean and standard deviation is computed on only the training data, 
    then used to normalize train and test */
    auto mean   = compute_mean(X_train);
    auto stddev = compute_std_dev(X_train, mean);

    normalize(X_train, mean, stddev);
    auto [X_test, y_test] = load_dataset("test");

    normalize(X_test, mean, stddev);
    auto X_test_reshaped = convert_to_images(X_test, 3, 32, 32);

    // Ensure features are always 32 x 32 x 3 = 3072 pixels
    if (X_train.size() != 3072) {
        cerr << "Error!: Unexpected feature size. Expected 3072, got " << X_train.size() << endl;
        return 1;
    }

    auto images = convert_to_images(X_train, image_channels, image_height, image_width); // Reshape from (3072) --> (32, 32, 3)
    auto labels = one_hot_encode(y_train, num_classes); // Converts [10000] vector to [10000, 10] matrix

    cout << "Initializing CNN model...\n";
    CNN cnn(image_channels, image_height, image_width, filter_size, num_filters, num_classes);

    cout << "Training the model...\n";
    CNNTrainer trainer(cnn, images, labels, learning_rate, batch_size, epochs);
    trainer.train();


    // Run the trained model on separate test data to test for generalization
    vector<int> predictions;
    for (const auto& img : X_test_reshaped) {
        int pred = cnn.predict(img);
        predictions.push_back(pred);
    }

    cout << "Predictions: " << predictions.size() << ", Ground truth: " << y_test.size() << endl;

    print_performance_summary(predictions, y_test);

    return 0;
}
