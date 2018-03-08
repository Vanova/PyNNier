import data
import sys
import numpy as np
import metrics.metrics as metr
import utils.plotters as ut_plt


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


def accuracy(ref, pred):
    """
    Return micro F1 error on test data
    :ref: reference, 2D array
    :pred: prediction, 2D array
    """
    # p = np.argmax(pred, axis=1)
    # p = cf.step(pred, threshold=0.5)
    # return metr.micro_f1(y_true=ref, y_pred=p, accuracy=False)
    return metr.pooled_accuracy(y_true=ref, y_pred=pred)


if __name__ == '__main__':
    from utils import mnist_loader

    data_type = 'mnist'

    if data_type == 'toy':
        num_examples = 1000
        output_dim = 12
        iterations = 1000
        x, y = data.generate_dataset(num_examples=num_examples, output_dim=output_dim)
    elif data_type == 'mnist':
        X_train, Y_train, X_val, Y_val, X_test, Y_test = mnist_loader.load_matrices()
        print("MNIST data is loaded...")
        epoches = 100
        batch_size = 10
        learn_rate = 3.  # 3.0
        nsamples, input_dim = X_train.shape
        layer_1_dim = 32
        layer_2_dim = 16
        nsamples, output_dim = Y_train.shape

    layer_1 = Layer(input_dim, layer_1_dim, sigmoid, sigmoid_out2deriv)
    layer_2 = Layer(layer_1_dim, layer_2_dim, sigmoid, sigmoid_out2deriv)
    layer_3 = Layer(layer_2_dim, output_dim, sigmoid, sigmoid_out2deriv)

    total_acc, total_error = [], []
    for i in range(epoches):
        error = 0
        for batch_i in range(int(len(X_train) / batch_size)):
            batch_x = X_train[(batch_i * batch_size): (batch_i + 1) * batch_size]
            batch_y = Y_train[(batch_i * batch_size): (batch_i + 1) * batch_size]

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

            error += np.linalg.norm(layer_3_delta) / batch_size
        if (i % 5 == 0):
            print("\rIter:" + str(i) + " error:" + str(error))

        layer_1_out = layer_1.forward(X_val)
        layer_2_out = layer_2.forward(layer_1_out)
        layer_3_out = layer_3.forward(layer_2_out)
        acc = accuracy(ref=Y_val, pred=layer_3_out)
        total_acc.append(acc)
        print(total_acc[-1])
    ut_plt.show_curves([total_acc],
                       legend=['val error acc'],
                       labels=["# of epochs", "value, %"],
                       title='Error accuracy, sigmoid scores')
    # ut_plt.show_curves([total_cost_1, total_cost_2],
    #                    legend=['syn_error_1', 'syn_error_2'],
    #                    labels=["# of epochs", "error value"],
    #                    title='Error, sigmoid scores')