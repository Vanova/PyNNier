"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

import random
import numpy as np
import math

class MifNetwork(object):
    def __init__(self, sizes, alpha=1.0, beta=0.0):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # progress report
        self.mif_smooth_progress = []
        self.mif_discrete_progress = []
        self.alpha = alpha
        self.beta = beta
        # for discrete mF1
        self.num_positive_lbls = 0
        self.correct = 0
        self.false_alarm = 0
        # for smooth mF1
        self.smooth_correct = 0.0
        self.smooth_false_alarm = 0.0
        self.progress_step = 10000


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def evaluate(self, test_data):
        """Calculate discrete MicroF1 value."""

    # """Return the number of test inputs for which the neural
    # network outputs the correct result. Note that the neural
    # network's output is assumed to be the index of whichever
    #     neuron in the final layer has the highest activation."""
    #     test_results = [(np.argmax(self.feedforward(x)), y)
    #                     for (x, y) in test_data]
    #     return sum(int(x == y) for (x, y) in test_results)


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Mini-batch stochastic gradient descent.
        The ``training_data`` is a list of tuples ``(x, y)``:
        the training inputs and the desired outputs.
        If ``test_data`` is provided then the network will be evaluated
        against the test data after each epoch for tracking training progress."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            count = 0
            for mini_batch in mini_batches:
                self.__update_mini_batch(mini_batch, eta)
                count += 1
                if count > self.progress_step:
                    print("batch {0}: loss = {1}".format(count, self.__cost_value()))
            smoothF1 = self.__cost_value()
            self.mif_smooth_progress.append(smoothF1)

            if test_data:
                mF1 = self.evaluate(test_data)
                self.mif_discrete_progress.append(mF1)
                print("Epoch {0}: microF1 = {1}, loss = {3}".format(j, mF1, smoothF1))
            else:
                print("Epoch {0} complete, loss = {1}".format(j, smoothF1))
        return (self.mif_discrete_progress, self.mif_smooth_progress)


    def __update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:  # for every training sample x
            # 1. propagate the sample,
            # get activations on every layer
            net_acts = self.__propagate(x)
            # 2. normalize network output signal
            norm_out_acts = self.__normalize_net_out(net_acts)
            # 3. backprop
            delta_nabla_b, delta_nabla_w = self.__backprop(norm_out_acts, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


    def __propagate(self, x):
        """Return activations on every layer of the network"""
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            activations.append(activation)
        return activations


    def __normalize_net_out(self, a):
        """Normalize the last network activation, i.e. output"""
        last = a[-1]
        last_e = np.exp(last)
        sum = np.sum(last_e)
        a[-1] = (last_e / sum).tolist()
        return a


    def __backprop(self, net_acts, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # backward pass
        # TODO fix here
        # loss = self.cost_value(net_out[-1], y)
        # calculate network error signal
        delta = self.__cost_derivative(net_acts[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, net_acts[-2].transpose())

        # start from the output of network, last layer l=-1
        for l in xrange(2, self.num_layers):
            a = net_acts[-l]
            sp = a * (1 - a)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, net_acts[-l - 1].transpose())
        return (nabla_b, nabla_w)


    def __cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial F_x /
        \partial z for the output activations."""
        diff = []  # result
        np = 0  # TP + FN
        # discrete thresholded counters
        tp = 0
        fp = 0
        # smooth approximation
        smooth_tp = 0.0
        smooth_fp = 0.0
        # derivative of loss
        dl = []
        num_ftrs = len(output_activations)
        ebeta = math.exp(-self.beta)

        for fId in xrange(num_ftrs):
            dk = (num_ftrs - 1) * output_activations[fId] / (1.0 - output_activations[fId])
            l = 1.0 / (1.0 + pow(dk, self.alpha) * ebeta)

            # smooth FN, FP, TP using loss function
            if y[fId] > 0:
                # count all positive labels: np = |C_k| = TP_k + FN_k
                np += 1
                smooth_tp += 1.0 - l
                if l < 0.5: tp += 1
            else:
                smooth_fp += 1.0 - l
            dl.append(self.alpha * l * (1.0 - l))

        # sum of rows of Jacobian matrix dl/dz
        for fId in xrange(num_ftrs):
            sum_dl = -dl[fId] / (1.0 - output_activations[fId])
            for xId in xrange(num_ftrs):
                sum_dl += dl[xId] * output_activations[fId] / (1.0 - output_activations[xId])
                dsigma = output_activations[fId] * (1.0 - output_activations[fId])
                diff[fId] = dsigma * sum_dl

        # the gradient of mF1 objective function
        a2 = (smooth_tp + smooth_fp + np) * (smooth_tp + smooth_fp + np)
        # if x_i in C_k
        scale_pos = 2.0 * (smooth_fp + np) / a2
        # if x_i not in C_k
        scale_neg = -2.0 * smooth_tp / a2
        for fId in xrange(num_ftrs):
            if y[fId] > 0:
                diff[fId] *= scale_pos
            else:
                diff[fId] *= scale_neg

        # for discrete micro F1
        self.num_positive_lbls += np
        self.correct += tp
        self.false_alarm += fp
        # for smoothed micro F1
        self.smooth_correct += smooth_tp
        self.smooth_false_alarm += smooth_fp

        return diff

    def __cost_value(self):
        """Return the smoothed MicroF1 value."""
        return 100.0 - 200.0 * self.smooth_correct / \
                       (self.smooth_correct + self.num_positive_lbls + self.smooth_false_alarm)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == '__main__':
    from utils import mnist_loader

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("MNIST data is loaded...")

    # MSE network with sigmoid output layer
    epochs = 30
    mini_batch = 10
    learn_rate = 3.0
    net = MifNetwork([784, 30, 10])
    err, loss = net.SGD(training_data, epochs, mini_batch, learn_rate, test_data=test_data)
    print(err)
    print(loss)
