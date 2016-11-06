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
import math
import time
import json
import numpy as np
from functions import metrics
import ann.cost_functions as cf


class MifNetwork(object):
    TRUE_LABEL = 1

    def __init__(self, sizes, alpha=10.0, beta=0.0):
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
        self.alpha = alpha
        self.beta = beta
        np.random.seed(888)
        self.default_weight_initializer()
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


    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = cf.sigmoid(np.dot(w, a) + b)
        return a


    def evaluate(self, test_data):
        """Calculate discrete MicroF1 value.
        Input: tuple list of ndarrays (data, target),
        targets should be binary vector"""
        net_predicts = [self.feedforward(xy[0]) for xy in test_data]
        refs = [xy[1] for xy in test_data]
        # threshold net output
        pred_labs = [cf.step(x, self.threshold) for x in net_predicts]
        return metrics.micro_f1(refs, pred_labs, False)


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
                print("Epoch {0}: F1_error_tst = {1}, F1_smooth_loss = {2}".format(j, mF1, smoothF1))
            else:
                print("Epoch {0} complete, loss = {1}".format(j, smoothF1))
        return (self.mif_discrete_progress, self.mif_smooth_progress)


    def __update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        data, lab = zip(*mini_batch)
        # forward data through the net
        activations, lab_pred = self.__propagate_on_batch(data)
        # back propagate and calculate gradient on the whole batch
        nabla_b, nabla_w = self.__backprop_on_batch(activations, lab_pred, lab)

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


    def __propagate_on_batch(self, mini_batch):
        """
        :return: list, TODO fix  every element is a 2D array with all layers activations
        for current sample in batch
        """
        acts = []
        predicts = []
        for x in mini_batch:  # for every training sample x
            # propagate the sample,
            # get activations on every layer
            net_acts = self.__propagate(x)
            acts.append(net_acts)
            predicts.append(net_acts[-1])
        return acts, predicts


    def __propagate(self, x):
        """Return activations on every layer of the network for sample x
        :param x: column np array
        """
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            activation = cf.sigmoid(z)
            activations.append(activation)
        return activations


    def __backprop_on_batch(self, batch_acts, lab_pred, lab):
        """
        Calculate back propagation on the mini-batch
        :param batch_acts: list of lists with activation arrays on every net layer
        :param lab_pred: list of arrays with net outputs,
        just a copy of the last layer activations
        :param lab: data labels
        :return: gradient DF/DW on the mini batch
        """
        ### calculate network error signal
        batch_delta = self.__cost_derivative(lab_pred, lab)

        # gradient on the mini batch
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # partial derivative on every sample
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]

        for delta, acts in zip(batch_delta, batch_acts):
            delta_nabla_b[-1] = delta
            delta_nabla_w[-1] = np.dot(delta, acts[-2].transpose())
            # start from the output of network, last layer l=-1
            for l in xrange(2, self.num_layers):
                a = acts[-l]
                sp = a * (1 - a)
                delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
                delta_nabla_b[-l] = delta
                delta_nabla_w[-l] = np.dot(delta, acts[-l - 1].transpose())
            # sum up the gradient
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        return (nabla_b, nabla_w)


    def __cost_derivative(self, lab_pred, lab):
        """Return the list of partial derivatives \partial F_x /\partial z
        for the output activations.
        F = 2TP/(FP + TP + npos),
        npos = TP + FN - number of positive samples
        :param lab_pred: network output on batch, list of arrays
        :return diff: list of gradients on each sample prediction
        """
        npos = 0
        # discrete thresholded counters
        # tp, fp = 0, 0
        # smooth approximation
        smooth_tp, smooth_fp = 0.0, 0.0
        diff = []
        nclass, _ = lab_pred[0].shape
        dl = np.array((nclass, 1))
        ebeta = math.exp(-self.beta)

        for p, y in zip(lab_pred, lab):
            ### normalization
            sum = np.sum(np.exp(p), axis=0)
            normp = np.exp(p) / sum
            ### strategy function
            dk = (nclass - 1) * normp / (1.0 - normp)
            l = 1.0 / (1.0 + np.power(dk, self.alpha) * ebeta)

            npos += (y == 1).sum()
            smooth_tp += np.sum((1.0 - l) * y, axis=0)[0]
            smooth_fp += np.sum((1.0 - l) * np.logical_not(y), axis=0)[0]

            Dl = self.alpha * l * (1.0 - l)
            ### sum of rows of Jacobian matrix dl/dz
            # TODO check this inference
            off_diag = np.empty((nclass, 1))
            diag = Dl / (1.0 - normp)
            sd = np.sum(diag, axis=0)
            off_diag.fill(sd[0])
            #off_diag.fill(0.0)
            sumJ = (off_diag * normp - diag) * p * (1.0 - p)
            diff.append(sumJ)

        ### the gradient of mF1 objective function
        a2 = (smooth_tp + smooth_fp + npos) * (smooth_tp + smooth_fp + npos)
        # if x_i in class C_k
        scale_pos = 2.0 * (smooth_fp + npos) / a2
        # if x_i not in class C_k
        scale_neg = -2.0 * smooth_tp / a2
        for i in xrange(len(diff)):
            # masked multiplication
            pos = diff[i] * scale_pos * lab[i]
            neg = diff[i] * scale_neg * np.logical_not(lab[i])
            diff[i] = pos + neg

        # for discrete micro F1
        # self.correct += tp
        # self.false_alarm += fp
        # micro F1 progress
        self.num_positive_lbls += npos
        self.smooth_correct += smooth_tp
        self.smooth_false_alarm += smooth_fp
        return diff

    def __cost_value(self):
        """Return the smoothed MicroF1 value."""
        return 100.0 - 200.0 * self.smooth_correct / \
                       (self.smooth_correct + self.num_positive_lbls + self.smooth_false_alarm)

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = MifNetwork(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


if __name__ == '__main__':
    from lib import mnist_loader
    from lib import toy_loader

    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data, validation_data, test_data = toy_loader.load_data(n_features=3)
    print("MNIST data is loaded...")
    epochs = 10
    mini_batch = 5
    learn_rate = 0.1
    architecture = [3, 10, 2]
    # architecture = [784, 30, 10]
    net2 = MifNetwork(architecture)
    print("MFoM micro F1 training...")
    start_time = time.time()
    err, loss = net2.SGD(training_data, epochs, mini_batch, learn_rate, test_data=test_data)
    end_time = time.time()
    total_time = end_time - start_time
    print(err)
    print(loss)
    print("Time: " + str(total_time))
