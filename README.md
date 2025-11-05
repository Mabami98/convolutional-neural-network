# Convolutional Neural Network for Image Classification

A simple convolutional neural network (CNN) implemented from scratch in C++ for image classification on CIFAR-10-style datasets.

## Features

- Forward- and backward pass with gradient updates
- Trained using Stochastic Gradient Descent (SGD)
- Training + testing pipeline
- Evaluation with per-class accuracy

## Model Overview

Each input image \( X \) is a 3D tensor of shape \( 32 \times 32 \times 3 \).

The CNN applies:

1. A set of convolutional filters \( F_i \), followed by ReLU:
   \[
   H_i = \text{ReLU}(X * F_i) \quad \text{for } i = 1, \dots, n_f
   \]
2. Concatenate and flatten:
   \[
   \mathbf{h} = \text{flatten}(H_1, H_2, \dots, H_{n_f})
   \]
3. Fully connected layer with ReLU:
   \[
   \mathbf{x}_1 = \text{ReLU}(W_1 \mathbf{h} + \mathbf{b}_1)
   \]
4. Output logits and softmax:
   \[
   \mathbf{s} = W_2 \mathbf{x}_1 + \mathbf{b}_2 \\
   \mathbf{p} = \text{Softmax}(\mathbf{s})
   \]

Loss is computed using cross-entropy:
\[
\mathcal{L} = -\sum_{c} y_c \log(p_c)
\]

## Build Instructions

```bash
make
./main
make clean
