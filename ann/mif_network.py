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
import time
import utils.mnist_loader as mload


class MifNetwork(object):
    def __init__(self, sizes, alpha=100.0, beta=0.0):
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
        self.alpha = alpha
        self.beta = beta
        # for discrete mF1
        self.threshold = 0.5
        self.num_positive_lbls = 0
        self.correct = 0
        self.false_alarm = 0
        # for smooth mF1
        self.smooth_correct = 0.0
        self.smooth_false_alarm = 0.0
        # progress report
        self.progress_step = 1000
        self.mif_smooth_progress = []
        self.mif_discrete_progress = []


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def evaluate(self, test_data):
        """Calculate discrete MicroF1 value.
        Input: tuple list of ndarrays (data, target),
        targets should be binary vector"""
        net_data = [(self.feedforward(x), y)
                    for x, y in test_data]
        net_predicts = [xy[0] for xy in net_data]
        refs = [xy[1] for xy in net_data]
        # threshold net output
        pred_labs = [step(x, self.threshold) for x in net_predicts]
        return micro_f1(refs, pred_labs, False)


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
                if divmod(count, self.progress_step)[1] == 0:
                    print("batch {0}: loss = {1}".format(count, self.__cost_value()))
            smoothF1 = self.__cost_value()
            self.mif_smooth_progress.append(smoothF1)

            if test_data:
                mF1 = self.evaluate(test_data)
                self.mif_discrete_progress.append(mF1)
                print("Epoch {0}: microF1 = {1}, loss = {2}".format(j, mF1, smoothF1))
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
        a[-1] = last_e / sum
        return a


    def __backprop(self, net_acts, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # backward pass
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
        npos = 0  # TP + FN
        # discrete thresholded counters
        tp = 0
        fp = 0
        # smooth approximation
        smooth_tp = 0.0
        smooth_fp = 0.0
        # derivative of loss
        dl = []
        num_ftrs = len(output_activations)
        diff = np.empty(y.shape)  # result
        ebeta = math.exp(-self.beta)

        for fId in xrange(num_ftrs):
            dk = (num_ftrs - 1) * output_activations[fId][0] / (1.0 - output_activations[fId][0])
            l = 1.0 / (1.0 + pow(dk, self.alpha) * ebeta)

            # smooth FN, FP, TP using loss function
            if y[fId] > 0:
                # count all positive labels: np = |C_k| = TP_k + FN_k
                npos += 1
                smooth_tp += 1.0 - l
                if l < self.threshold: tp += 1
            else:
                smooth_fp += 1.0 - l
                if l < self.threshold: fp += 1
            dl.append(self.alpha * l * (1.0 - l))

        # sum of rows of Jacobian matrix dl/dz
        for fId in xrange(num_ftrs):
            sum_dl = -dl[fId] / (1.0 - output_activations[fId][0])
            for xId in xrange(num_ftrs):
                sum_dl += dl[xId] * output_activations[fId][0] / (1.0 - output_activations[xId][0])
            dsigma = output_activations[fId][0] * (1.0 - output_activations[fId][0])
            diff[fId][0] = dsigma * sum_dl

        # the gradient of mF1 objective function
        a2 = (smooth_tp + smooth_fp + npos) * (smooth_tp + smooth_fp + npos)
        # if x_i in C_k
        scale_pos = 2.0 * (smooth_fp + npos) / a2
        # if x_i not in C_k
        scale_neg = -2.0 * smooth_tp / a2
        for fId in xrange(num_ftrs):
            if y[fId] > 0:
                diff[fId][0] *= scale_pos
            else:
                diff[fId][0] *= scale_neg

        # for discrete micro F1
        self.num_positive_lbls += npos
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


def step(a, threshold=0.0):
    """Heaviside step function:
    a < threshold = 0, else 1.
    Input: array
    Output: array"""
    res = [[0] if x < threshold else [1] for x in a ]
    return np.array(res) # need column


def micro_f1(refs, predicts, accuracy=True):
    """Input: binary integer list of arrays"""
    assert(len(refs) == len(predicts))
    neg_r = np.logical_not(refs)
    neg_p = np.logical_not(predicts)
    tp = np.sum(np.logical_and(refs, predicts) == True)
    fp = np.sum(np.logical_and(neg_r, predicts) == True)
    fn = np.sum(np.logical_and(refs, neg_p) == True)
    f1 = 100.0 * 2.0 * tp / (2.0*tp + fp + fn)
    return accuracy and f1 or 100.0 - f1


if __name__ == '__main__':
    from utils import mnist_loader
    from ann.network import Network

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("MNIST data is loaded...")
    epochs = 10
    mini_batch = 10
    learn_rate = 0.001
    architecture = [784, 30, 10]
    net2 = MifNetwork(architecture)
    print("MFoM micro F1 training...")
    start_time = time.time()
    err, loss = net2.SGD(training_data, epochs, mini_batch, learn_rate, test_data=test_data)
    end_time = time.time()
    total_time = end_time - start_time
    print(err)
    print(loss)
    print("Time: " + str(total_time))
