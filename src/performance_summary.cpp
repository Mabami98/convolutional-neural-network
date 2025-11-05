#include "../include/performance_summary.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace std;

void print_performance_summary(const vector<int>& predictions,
                               const vector<int>& ground_truth) {

    const vector<string> class_names = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };

    const int num_classes = class_names.size();
    std::vector<int> correct_per_class(num_classes, 0);
    std::vector<int> total_per_class(num_classes, 0);

    for (size_t i = 0; i < predictions.size(); ++i) {
        int true_label = ground_truth[i];
        int pred_label = predictions[i];
        total_per_class[true_label]++;
        if (pred_label == true_label)
            correct_per_class[true_label]++;
    }

    cout << "\nPerformance Summary:\n";
    cout << left << setw(15) << "Class"
              << right << setw(12) << "Accuracy"
              << right << setw(18) << "Correct / Total" << "\n";
    cout << '-----------------------------------------' << "\n";

    int total_correct = 0;
    int total_count = 0;

    for (int i = 0; i < num_classes; ++i) {
        int correct = correct_per_class[i];
        int total = total_per_class[i];
        float accuracy = (total > 0) ? static_cast<float>(correct) / total : 0.0f;

        total_correct += correct;
        total_count += total;

        cout << left << setw(15) << class_names[i]
                  << right << setw(9) << fixed << setprecision(2)
                  << accuracy * 100 << " %"
                  << right << setw(12) << correct << " / " << total << "\n";
    }

    float overall = static_cast<float>(total_correct) / total_count;
    std::cout << "\nOverall Accuracy: " << std::fixed << std::setprecision(2)
              << overall * 100 << "% (" << total_correct << "/" << total_count << ")\n";
}
