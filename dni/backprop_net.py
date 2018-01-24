"""
Simple network implementation
using backprop equations directly, without OOP
"""
import data
import sys
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


num_examples = 1000
output_dim = 12
iterations = 1000
x, y = data.generate_dataset(num_examples=num_examples, output_dim=output_dim)

batch_size = 10
alpha = 0.1
# layer_neurons = [len(x[0]), 128, 64, len(y[0])]
input_dim = len(x[0])
layer_1_dim = 128
layer_2_dim = 64
output_dim = len(y[0])

weights_0_1 = np.random.randn(input_dim, layer_1_dim) * 0.2 - 0.1
bias_0_1 = np.random.randn(layer_1_dim) * 0.2 - 0.1
weights_1_2 = np.random.randn(layer_1_dim, layer_2_dim) * 0.2 - 0.1
bias_1_2 = np.random.randn(layer_2_dim) * 0.2 - 0.1
weights_2_3 = np.random.randn(layer_2_dim, output_dim) * 0.2 - 0.1
bias_2_3 = np.random.randn(output_dim) * 0.2 - 0.1

for iter in range(iterations):
    error = 0
    for batch_i in range(int(len(x) / batch_size)):
        batch_x = x[(batch_i * batch_size):(batch_i + 1) * batch_size]
        batch_y = y[(batch_i * batch_size):(batch_i + 1) * batch_size]

        # forward pass
        layer_0 = batch_x
        layer_1 = sigmoid(layer_0.dot(weights_0_1) + bias_0_1)
        layer_2 = sigmoid(layer_1.dot(weights_1_2) + bias_1_2)
        layer_3 = sigmoid(layer_2.dot(weights_2_3) + bias_2_3)
        # backward pass
        layer_3_delta = (layer_3 - batch_y) * layer_3 * (1 - layer_3)
        layer_2_delta = layer_3_delta.dot(weights_2_3.T) * layer_2 * (1 - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * layer_1 * (1 - layer_1)

        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)
        bias_0_1 -= np.mean(layer_1_delta, axis=0)
        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        bias_1_2 -= np.mean(layer_2_delta, axis=0)
        weights_2_3 -= alpha * layer_2.T.dot(layer_3_delta)
        bias_2_3 -= np.mean(layer_3_delta, axis=0)

        error += (np.sum(np.abs(layer_3_delta)))

    sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
    if (iter % 100 == 99):
        print("")
