"""
~~~~~~~~~~~~~~
Based on Michael Nielsen: Network2.py
"""

import json
import random
import sys
import numpy as np
import cost_functions as cf
from metrics import metrics
import copy

np.random.seed(777)
random.seed(777)


#### Main Network class
class MatrixNetwork(object):
    def __init__(self, sizes, threshold_f1=0.5, cost=cf.QuadraticCost):
        """
        :param sizes: list contains the number of neurons in the respective
        layers of the network, e.g. [2, 3, 1].
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.threshold_f1 = threshold_f1
        # random weights and bias initialization
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        # rows = #_of_samples, columns = dim
        self.biases = [np.random.randn(1, y) for y in self.sizes[1:]]
        # rows = input_dim, columns = output_dim
        self.weights = [np.random.randn(x, y) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """
        :return: the output of the network
        """
        for b, w in zip(self.biases, self.weights):
            a = cf.sigmoid(np.dot(a, w) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            is_list_weights=False,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        Monitor the cost and accuracy on either the evaluation data or
         the training data, by setting the appropriate flags.
        :param training_data: list of tuples (x, y),
        x are training inputs, y are  desired outputs
        :param lmbda: regularization parameter
        :param evaluation_data: validation or test set
        :return: lists with per-epoch costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data. The lists
        are empty if the corresponding flag is not set.
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        list_weights = []
        for j in xrange(epochs):
            random.shuffle(training_data)
            data_batches, label_batches = create_minibatches(training_data, mini_batch_size)
            for X, Y in zip(data_batches, label_batches):
                self._update_mini_batch(X, Y, eta,
                                        lmbda, len(training_data))
            print "Epoch %s training complete" % j
            # save network weights optimization, e.g. in order to plot
            if is_list_weights:
                # TODO append biases as well
                list_weights.append(copy.deepcopy(self.weights))
            # monitoring
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=False)
                training_accuracy.append(accuracy)
                print "Accuracy on training data (discrete F1): {}".format(accuracy)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=False)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=True)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data (discrete F1): {}".format(accuracy)
            print
        return evaluation_cost, evaluation_accuracy, \
               training_cost, training_accuracy, list_weights

    def _update_mini_batch(self, data_batch, labs_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``data and labs`` are two arrayes, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        # forward data through the net
        affines, activations = self._propagate(data_batch)
        # back propagate and calculate gradient on the whole batch
        nabla_b, nabla_w = self._backprop(affines, activations, labs_batch)

        self.biases = [b - eta * nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [(1 - eta * (lmbda / n)) * w - eta * nw
                        for w, nw in zip(self.weights, nabla_w)]

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
        """ Input: x, y - matrices (batch_size, dim1)
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        ### calculate network error signal
        delta = (self.cost).delta(affines[-1], activations[-1], labels)
        ###
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

    def accuracy(self, data, convert=False):
        """Return micro F1 error on test data"""
        predicts = []
        refs = []
        for x, y in data:
            # thresholded prediction
            p = cf.step(self.feedforward(x.T), self.threshold_f1)
            predicts.append(p)
            refs.append(y.T)
        return metrics.micro_f1(refs=refs, predicts=predicts, accuracy=False)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            x = x.transpose()
            y = y.transpose()
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)
        # cost += 0.5 * (lmbda / len(data)) * \
        #         sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost / len(data)

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = MatrixNetwork(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### Miscellaneous metrics
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


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


if __name__ == "__main__":
    import time
    from utils import toy_loader
    from sklearn import preprocessing
    from utils.plotters import show_curves

    feature_dim = 2
    n_classes = 2
    train_data, validation_data, test_data = toy_loader.load_data(n_features=feature_dim, n_classes=n_classes,
                                                                  scaler=preprocessing.StandardScaler())
    print("Toy data is loaded...")
    epochs = 100
    mini_batch = 5
    learn_rate = 0.01
    architecture = [feature_dim, n_classes]
    net = MatrixNetwork(architecture)
    # training
    start_time = time.time()
    eval_cost, eval_acc, tr_cost, tr_acc, _ = net.SGD(train_data, epochs, mini_batch,
                                                      learn_rate, evaluation_data=validation_data,
                                                      monitor_evaluation_cost=True,
                                                      monitor_evaluation_accuracy=True,
                                                      monitor_training_cost=True,
                                                      monitor_training_accuracy=True)
    end_time = time.time()
    print("Time: " + str(end_time - start_time))
    print(eval_acc[-1])  # 2.36220472441
    print(eval_cost[-1])  # 0.0521401561191
    print(tr_cost[-1])  # 0.0554160574567
    show_curves([eval_cost, tr_cost],
                legend=["evaluation cost", "training cost"],
                labels=["# of epochs", "value"],
                title="MSE cost function")
    show_curves([eval_acc, tr_acc],
                legend=["evaluation acc", "training acc"],
                labels=["# of epochs", "value, %"],
                title="Micro F1 value, sigmoid scores")
