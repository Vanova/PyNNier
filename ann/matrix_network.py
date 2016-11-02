"""
~~~~~~~~~~~~~~
Based on Michael Nielsen: Network2.py
"""

import json
import random
import sys
import numpy as np
import cost_functions as cf

np.random.seed(777)
# random.seed(777)


#### Main Network class
class MatrixNetwork(object):
    def __init__(self, sizes, cost=cf.QuadraticCost):
        """
        :param sizes: list contains the number of neurons in the respective
        layers of the network, e.g. [2, 3, 1].
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
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
        for j in xrange(epochs):
            random.shuffle(training_data)
            data_batches, label_batches = create_minibatches(training_data, mini_batch_size)
            for X, Y in zip(data_batches, label_batches):
                self.update_mini_batch(
                    X, Y, eta, lmbda, len(training_data))
            print "Epoch %s training complete" % j
            # monitoring
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=False)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(
                    accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=False)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=True)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data, convert=True), n_data)
            print
        return evaluation_cost, evaluation_accuracy, \
               training_cost, training_accuracy

    def update_mini_batch(self, data, labs, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``data and labs`` are two arrayes, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b, nabla_w = self.backprop(data, labs)
        self.biases = [b - eta * nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [(1 - eta * (lmbda / n)) * w - eta * nw
                        for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        """ Input: x, y - matrices (batch_size, dim1)
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(x, w) + b
            zs.append(z)
            activation = cf.sigmoid(z)
            activations.append(activation)
        # backward pass
        mini_batch_size = x.shape[0]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        delta = (self.cost).delta(zs[-1], activations[-1], y)
        # TODO fix here
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(activations[-2].transpose(), delta)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = cf.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        avg_nabla_b = [np.mean(nb, axis=0) for nb in nabla_b]
        avg_nabla_w = [nw/mini_batch_size for nw in nabla_w]
        return avg_nabla_b, avg_nabla_w

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x.transpose())), np.argmax(y.transpose()))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

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
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

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


#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def create_minibatches(data, mini_batch_size):
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
    from utils import mnist_loader
    from utils import toy_loader
    from sklearn import preprocessing

    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # print("MNIST data is loaded...")
    feature_dim = 3
    train_data, validation_data, test_data = toy_loader.load_data(n_tr=250, n_dev=50, n_tst=50,
                                                                  n_features=feature_dim, n_classes=2,
                                                                  scaler=preprocessing.StandardScaler())
    print("Toy data is loaded...")
    epochs = 10
    mini_batch = 5
    learn_rate = 0.1
    architecture = [feature_dim, 2]
    # net = Network(architecture)
    net = MatrixNetwork(architecture)
    # training
    start_time = time.time()
    eval_cost, eval_acc, tr_cost, tr_acc = net.SGD(train_data, epochs, mini_batch,
                                                   learn_rate, evaluation_data=validation_data,
                                                   monitor_evaluation_cost=True,
                                                   monitor_evaluation_accuracy=True,
                                                   monitor_training_cost=True)
    end_time = time.time()
    print("Time: " + str(end_time - start_time))
    print(eval_acc)
    print(eval_cost[-1])  # 0.05522
    print(tr_cost[-1])  # 0.05845
