"""
~~~~~~~~~~~~~~
Based on Michael Nielsen
"""

import copy
import json
import random
import numpy as np
from sklearn.metrics import mean_squared_error
import nonlinearity as nonl
import ann.lazy.cost_functions as cf
from metrics import metrics

np.random.seed(777)
random.seed(777)

class Network(object):
    def __init__(self, sizes, threshold_f1=0.5):
        """
        :param sizes: list contains the number of neurons in the respective
        layers of the network, e.g. [2, 3, 1].
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.threshold_f1 = threshold_f1
        # random weights and bias initialization
        self.default_weight_initializer()
        self.eval_err_progress = []
        self.loss_tr_progress = []

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """
        Input: data 2D array,  Dimension x N samples
        Return the output of the network (2D array),
        forward data through all layers.
        """
        for b, w in zip(self.biases, self.weights):
            a = nonl.sigmoid(np.dot(w, a) + b)
        return a

    def propagate(self):
        """Forward the data array trough the network.
        Input: data 2D array, N samples x Dimension
        Output the activations only from the last layer.
        """
        pass

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, is_list_weights=False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        list_weights = []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            mean_loss = 0.0
            for mini_batch in mini_batches:
                mean_loss += self.update_mini_batch(mini_batch, eta)
            mean_loss = mean_loss / len(mini_batches)
            self.loss_tr_progress.append(mean_loss)
            if is_list_weights:
                list_weights.append(copy.deepcopy(self.weights))

            if test_data:
                err = self.evaluate(test_data)
                true_pos = self.correct_count(test_data)
                self.eval_err_progress.append(err)
                print("Epoch {0}: F1_error_tst = {1}, MSE_loss = {2}".
                      format(j, err, mean_loss))
            else:
                print("Epoch {0} complete, loss = {1}".format(j, mean_loss))
        return self.eval_err_progress, self.loss_tr_progress, list_weights

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        batch_loss = 0.0
        for x, y in mini_batch:  # for every training sample
            delta_nabla_b, delta_nabla_w, loss_val = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            batch_loss += loss_val
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        return batch_loss / len(mini_batch)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = nonl.sigmoid(z)
            activations.append(activation)
        loss = self.total_cost(activations[-1], y)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * nonl.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = nonl.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w, loss)

    def correct_count(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        # count number of right answers
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate(self, test_data):
        """Return micro F1 error on test data"""
        predicts = []
        refs = []
        for x, y in test_data:
            # thresholded prediction
            p = cf.step(self.feedforward(x), self.threshold_f1)
            predicts.append(p)
            refs.append(y)
        return metrics.micro_f1(y_true=refs, y_pred=predicts, accuracy=False)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)

    def total_cost(self, output_activations, y):
        """Return the MSE loss function value between
         the network output and target label y."""
        return mean_squared_error(y_true=y, y_pred=output_activations)

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        # TODO fix "cost": str(self.cost.__name__)
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
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


if __name__ == '__main__':
    # import time
    # from utils import mnist_loader
    #
    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # print("MNIST data is loaded...")
    # epochs = 10
    # mini_batch = 10
    # learn_rate = 3.0
    # net = Network([784, 30, 10])
    # # training
    # start_time = time.time()
    # f1, loss, _ = net.SGD(training_data, epochs, mini_batch, learn_rate, test_data=test_data)
    # end_time = time.time()
    # print("Time: " + str(end_time - start_time))
    # print(f1)
    # print(loss)
    # # save
    # file_net = "./data/experiment/nist/nist_epo_{0}_btch_{1}_lr_{2}". \
    #     format(epochs, mini_batch, learn_rate)
    # net.save(file_net)
    # # test load
    # net2 = load(file_net)
    # f1, loss, _ = net2.SGD(training_data, epochs, mini_batch, learn_rate, test_data=test_data)
    # print(f1)
    # print(loss)

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
    net = Network(architecture)
    # training
    start_time = time.time()
    f1_eval, tr_loss, _ = net.SGD(train_data, epochs, mini_batch,
                           learn_rate, test_data=validation_data)
    end_time = time.time()
    print("Time: " + str(end_time - start_time))
    print(f1_eval[-1])  # 3.87596899225
    print(tr_loss[-1])  # 0.0586984883838
    show_curves([tr_loss],
                legend=["training loss"],
                labels=["# of epochs", "value, %"],
                title="MSE cost function")
    show_curves([f1_eval],
                legend=["evaluation acc"],
                labels=["# of epochs", "value"],
                title="Micro F1 value")


