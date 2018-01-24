"""
Synthetic Gradient network, summation of two binary digits
"""
import sys
import numpy as np
import data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(out):
    return out * (1 - out)


class DNI(object):
    def __init__(self, input_dim, output_dim, nonlin, nonlin_deriv, alpha=0.1):

        self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
        self.bias = (np.random.randn(output_dim) * 0.2) - 0.1

        self.weights_0_1_synthetic_grads = (np.random.randn(output_dim, output_dim) * .2) - .1
        self.bias_0_1_synthetic_grads = (np.random.randn(output_dim) * .2) - .1

        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        self.alpha = alpha

    def forward_and_synthetic_update(self, input, update=True):

        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights) + self.bias)

        if (not update):
            return self.output
        else:
            self.synthetic_gradient = self.output.dot(self.weights_0_1_synthetic_grads) + self.bias_0_1_synthetic_grads
            self.weight_synthetic_gradient = self.synthetic_gradient * self.nonlin_deriv(self.output)

            self.weights -= self.input.T.dot(self.weight_synthetic_gradient) * self.alpha
            self.bias -= np.average(self.weight_synthetic_gradient, axis=0) * self.alpha

        return self.weight_synthetic_gradient.dot(self.weights.T), self.output

    def normal_update(self, true_gradient):
        grad = true_gradient * self.nonlin_deriv(self.output)

        self.weights -= self.input.T.dot(grad) * self.alpha
        self.bias -= np.average(grad, axis=0) * self.alpha

        return grad.dot(self.weights.T)

    def update_synthetic_weights(self, true_gradient):
        self.synthetic_gradient_delta = (self.synthetic_gradient - true_gradient)
        self.weights_0_1_synthetic_grads -= self.output.T.dot(self.synthetic_gradient_delta) * self.alpha
        self.bias_0_1_synthetic_grads -= np.average(self.synthetic_gradient_delta, axis=0) * self.alpha


num_examples = 1000
output_dim = 12
iterations = 10000
x, y = data.generate_dataset(num_examples=num_examples, output_dim=output_dim)

batch_size = 10
alpha = 0.0001

input_dim = len(x[0])
layer_1_dim = 128
layer_2_dim = 64
output_dim = len(y[0])

# define 3 layers network with synthetic gradients weights
layer_1 = DNI(input_dim, layer_1_dim, sigmoid, sigmoid_deriv, alpha)
layer_2 = DNI(layer_1_dim, layer_2_dim, sigmoid, sigmoid_deriv, alpha)
layer_3 = DNI(layer_2_dim, output_dim, sigmoid, sigmoid_deriv, alpha)

for iter in range(iterations):
    error = 0
    synthetic_error = 0

    for batch_i in range(int(len(x) / batch_size)):
        batch_x = x[(batch_i * batch_size):(batch_i + 1) * batch_size]
        batch_y = y[(batch_i * batch_size):(batch_i + 1) * batch_size]
        # forward pass
        _, layer_1_out = layer_1.forward_and_synthetic_update(batch_x)
        layer_1_delta, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
        # TODO debug
        layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out, False)
        # backward pass
        layer_3_delta = layer_3_out - batch_y
        layer_2_delta = layer_3.normal_update(layer_3_delta)
        layer_2.update_synthetic_weights(layer_2_delta)
        layer_1.update_synthetic_weights(layer_1_delta)

        # error += (np.sum(np.abs(layer_3_delta)))
        error += (np.sum(np.abs(layer_3_delta * layer_3_out * (1 - layer_3_out))))
        synthetic_error += (np.sum(np.abs(layer_2_delta - layer_2.synthetic_gradient)))
    if (iter % 100 == 99):
        sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error) + " Synthetic Loss:" + str(synthetic_error))
    if (iter % 10000 == 9999):
        print("")
