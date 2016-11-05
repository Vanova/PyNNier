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

np.random.seed(777)
random.seed(777)


class MFoMNetwork(object):
    TRUE_LABEL = 1

    def __init__(self, sizes, alpha=10.0, beta=0.0, cost=cf.MFoMCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.alpha = alpha
        self.beta = beta
        # random weights and bias initialization
        self.default_weight_initializer()
        self.cost = cost
        if self.cost == cf.MFoMCost:
            self.cost.alpha = alpha
            self.cost.beta = beta

        # for discrete mF1
        self.threshold = 0.5
        # self.num_positive_lbls = 0
        # self.correct = 0
        # self.false_alarm = 0
        # for smooth mF1
        # self.smooth_correct = 0.0
        # self.smooth_false_alarm = 0.0

    def default_weight_initializer(self):
        # rows = #_of_samples, columns = dim
        self.biases = [np.random.randn(1, y) for y in self.sizes[1:]]
        # rows = input_dim, columns = output_dim
        self.weights = [np.random.randn(x, y) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(1, y) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """
        :return: the outputs of the network only
        """
        for b, w in zip(self.biases, self.weights):
            a = cf.sigmoid(np.dot(a, w) + b)
        return a

    def accuracy(self, test_data):
        """Calculate discrete MicroF1 value.
        Input: tuple list of ndarrays (data, target),
        targets should be binary vector"""
        net_predicts = [self.feedforward(xy[0]) for xy in test_data]
        refs = [xy[1] for xy in test_data]
        # threshold net output
        pred_labs = [cf.step(x, self.threshold) for x in net_predicts]
        return metrics.micro_f1(refs, pred_labs, False)

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Mini-batch stochastic gradient descent.
        The ``training_data`` is a list of tuples ``(x, y)``:
        the training inputs and the desired outputs.
        If ``test_data`` is provided then the network will be evaluated
        against the test data after each epoch for tracking training progress."""
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            data_batches, label_batches = create_minibatches(training_data, mini_batch_size)
            for X, Y in zip(data_batches, label_batches):
                self._update_mini_batch(X, Y, eta,
                                        lmbda, len(training_data))
            print "Epoch %s training complete" % j

            # monitoring
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data (smooth F1): {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=False)
                training_accuracy.append(accuracy)
                print "Accuracy on training data (discrete F1): {} / {}".format(
                    accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=False)
                evaluation_cost.append(cost)
                print "Cost on evaluation data (smooth F1): {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=True)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data (discrete F1): {} / {}".format(
                    self.accuracy(evaluation_data, convert=True), n_data)
            print

        # evaluation_smooth_f1, evaluation_discrete_f1
        # training_smooth_f1, training_discrete_f1
        return evaluation_cost, evaluation_accuracy, \
               training_cost, training_accuracy

    def _update_mini_batch(self, data_batch, labs_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # forward data through the net
        affines, activations = self._propagate(data_batch)
        # back propagate and calculate gradient on the whole batch
        nabla_b, nabla_w = self._backprop(affines, activations, labs_batch)

        # TODO add momentum
        self.weights = [w - eta * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def _propagate(self, x):
        """
        Input: 2D array: rows = # of samples, columns = dim
        :return: list of activations from all hidden layers
         of the network
        """
        a = x
        acts = [x]  # list to store all the activations, layer by layer
        lin = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(a, w) + b
            lin.append(z)
            a = cf.sigmoid(z)
            acts.append(a)
        return lin, acts

    def _backprop(self, affines, activations, labels):
        """
        Calculate back propagation on the mini-batch
        :param activations: list of lists with activation arrays on every net layer
        :param lab_pred: list of arrays with net outputs,
        just a copy of the last layer activations
        :param labels: data labels
        :return: gradient DF/DW on the mini batch
        """
        ### calculate network error signal
        delta = (self.cost).delta(affines[-1], activations[-1], labels)
        ########
        mini_batch_size = activations[0].shape[0]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(activations[-2].transpose(), delta)

        for l in xrange(2, self.num_layers):
            z = affines[-l]
            sp = cf.sigmoid_prime(z)
            delta = np.dot(delta, self.weights[-l + 1].transpose()) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(activations[-l - 1].transpose(), delta)
        avg_nabla_b = [np.mean(nb, axis=0) for nb in nabla_b]
        avg_nabla_w = [nw / mini_batch_size for nw in nabla_w]
        return avg_nabla_b, avg_nabla_w

    def total_cost(self):
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
    net = MFoMNetwork(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def create_minibatches(data, mini_batch_size):
    # TODO fix format during loading, not training
    n = len(data)
    mini_batches = [
        data[k:k + mini_batch_size]
        for k in xrange(0, n, mini_batch_size)]
    data_batches = []
    label_batches = []
    for batch in mini_batches:
        x, y = zip(*batch)
        data_batches.append(np.array(x).squeeze())
        label_batches.append(np.array(y).squeeze())
    return data_batches, label_batches


if __name__ == '__main__':
    import time
    from utils import toy_loader
    from sklearn import preprocessing

    feature_dim = 3
    n_classes = 2
    train_data, validation_data, test_data = toy_loader.load_data(n_features=feature_dim, n_classes=n_classes,
                                                                  scaler=preprocessing.StandardScaler())
    print("Toy data is loaded...")
    epochs = 10
    mini_batch = 5
    learn_rate = 0.1
    architecture = [feature_dim, n_classes]
    net = MFoMNetwork(architecture)
    # training
    start_time = time.time()
    eval_cost, eval_acc, tr_cost, tr_acc = net.SGD(train_data, epochs, mini_batch,
                                                   learn_rate, evaluation_data=validation_data,
                                                   monitor_evaluation_cost=True,
                                                   monitor_evaluation_accuracy=True,
                                                   monitor_training_cost=True,
                                                   monitor_training_accuracy=True)
    end_time = time.time()
    print("Time: " + str(end_time - start_time))
    print(eval_acc)
    print(eval_cost[-1])  # 0.05727
    print(tr_cost[-1])  # 0.07007
