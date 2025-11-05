# Convolutional Neural Network for Image Classification

A simple convolutional neural network (CNN) implemented from scratch in C++ for image classification on CIFAR-10-style datasets.

## Features

- Forward- and backward pass with gradient updates
- Trained using Stochastic Gradient Descent (SGD)
- Training + testing pipeline
- Evaluation with per-class accuracy

## Build Instructions

```bash
make
./main
make clean


## Sample Output

Loading training data...
Loaded train set: (10000 images)
  X shape: 3072 x 10000
  y size:  10000
Normalizing training data...
Reshaping and encoding training data...
Initializing CNN model...
Training model...

Epoch 1/10 | Loss: 2.135 | Accuracy: 21.3%
Epoch 2/10 | Loss: 2.002 | Accuracy: 28.4%
...

Performance Summary:
Class           Accuracy      Correct / Total
---------------------------------------------
airplane        73.50 %       735 / 1000
automobile      80.10 %       801 / 1000
...
Overall Accuracy: 76.45% (7645/10000)
