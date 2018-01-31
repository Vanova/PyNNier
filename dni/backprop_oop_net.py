import data
import sys
import numpy as np


class Layer(object):
    def __init__(self, input_dim, output_dim, nonlin, nonlin_deriv):
        self.weights = 0.2 * np.random.randn(input_dim, output_dim) - 0.1
        self.bias = 0.2 * np.random.randn(output_dim) - 0.1
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        # init network layer
        self.input = np.empty(input_dim)
        self.output = np.empty(output_dim)
        self.weight_output_delta = np.empty(output_dim)

    def forward(self, input):
        """ Calculate forward network activated signal"""
        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights) + self.bias)
        return self.output

    def backward(self, output_delta):
        """ Network error signal
            output_delta: gradient from the previous layer
        """
        self.weight_output_delta = output_delta * self.nonlin_deriv(self.output)
        return self.weight_output_delta.dot(self.weights.T)

    def update(self, alpha=0.1):
        """ Update weights and biases of current layer"""
        self.weights -= alpha * self.input.T.dot(self.weight_output_delta)
        self.bias -= alpha * np.mean(self.bias, axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_out2deriv(x):
    return x * (1 - x)


num_examples = 1000
output_dim = 12
iterations = 1000
x, y = data.generate_dataset(num_examples=num_examples, output_dim=output_dim)

batch_size = 256
alpha = 0.1
# layer_neurons = [len(x[0]), 128, 64, len(y[0])]
input_dim = len(x[0])
layer_1_dim = 128
layer_2_dim = 64
output_dim = len(y[0])

layer_1 = Layer(input_dim, layer_1_dim, sigmoid, sigmoid_out2deriv)
layer_2 = Layer(layer_1_dim, layer_2_dim, sigmoid, sigmoid_out2deriv)
layer_3 = Layer(layer_2_dim, output_dim, sigmoid, sigmoid_out2deriv)

for i in range(iterations):
    error = 0
    for batch_i in range(int(len(x) / batch_size)):
        batch_x = x[(batch_i * batch_size): (batch_i + 1) * batch_size]
        batch_y = y[(batch_i * batch_size): (batch_i + 1) * batch_size]

        layer_1_out = layer_1.forward(batch_x)
        layer_2_out = layer_2.forward(layer_1_out)
        layer_3_out = layer_3.forward(layer_2_out)

        layer_3_delta = layer_3_out - batch_y
        layer_2_delta = layer_3.backward(layer_3_delta)
        layer_1_delta = layer_2.backward(layer_2_delta)
        layer_1.backward(layer_1_delta)

        layer_1.update()
        layer_2.update()
        layer_3.update()

        error += np.sum(np.abs(layer_3_delta * sigmoid_out2deriv(layer_3_out)))

    sys.stdout.write("\rIter:" + str(i) + " Loss:" + str(error))
    if i % 100 == 99:
        print("")
