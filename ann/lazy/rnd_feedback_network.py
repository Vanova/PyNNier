"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# ### Libraries
# Standard library
import random
import numpy as np
import nonlinearity as nonl
from sklearn.metrics import mean_squared_error


class RFANetwork(object):
    def __init__(self, sizes):
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
        self.rnd_feedback = [(np.random.rand(y, x) - 0.5) for x, y in zip(sizes[1:-1], sizes[2:])]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.loss_progress = []
        self.acc_progress = []


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = nonl.sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch"""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            loss = 0.0
            for mini_batch in mini_batches:
                loss += self.update_mini_batch(mini_batch, eta)
            mean_loss = loss / len(mini_batch)
            self.loss_progress.append(mean_loss)

            if test_data:
                err = self.evaluate(test_data)
                self.acc_progress.append(err)
                print("Epoch {0}: {1} / {2}, loss = {3}".format(j, err, n_test, mean_loss))
            else:
                print("Epoch {0} complete, loss = {1}".format(j, mean_loss))
        return (self.acc_progress, self.loss_progress)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        batch_loss = 0.0

        for x, y in mini_batch:
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
        # backward pass
        loss = self.cost_value(activations[-1], y)
        delta = self.cost_derivative(activations[-1], y) * nonl.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # do random feedback weighting first
        # start from last layer l=-1
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = nonl.sigmoid_prime(z)
            delta = np.dot(self.rnd_feedback[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w, loss)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)

    def cost_value(self, output_activations, y):
        """Return the MSE loss function value between
         the network output and target label y."""
        return mean_squared_error(y_true=y, y_pred=output_activations)


if __name__ == '__main__':
    from utils import mnist_loader

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("MNIST data is loaded...")
    # MSE network with sigmoid output layer
    epochs = 30
    mini_batch = 10
    learn_rate = 3.0
    net = RFANetwork([784, 30, 10])
    acc, loss = net.SGD(training_data, epochs, mini_batch, learn_rate, test_data=test_data)
    print(acc)
    print(loss)
