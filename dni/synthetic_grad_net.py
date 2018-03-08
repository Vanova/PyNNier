"""
Synthetic Gradient network, summation of two binary digits
"""
import numpy as np
import data
import ann.lazy.nonlinearity as nonl
import ann.lazy.cost_functions as cf
import metrics.metrics as metr
import utils.plotters as ut_plt


class DNI(object):
    def __init__(self, input_dim, output_dim, nonlin, nonlin_deriv, alpha=0.1):
        # ***
        # main network weights
        # ***
        self.weights = 0.2 * np.random.randn(input_dim, output_dim) - 0.1
        self.bias = 0.2 * np.random.randn(output_dim) - 0.1
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        # init network layer
        self.input = np.empty(input_dim)
        self.output = np.empty(output_dim)
        self.weight_output_delta = np.empty(output_dim)
        # ***
        # synthetic gradient net: linear regression
        # ***
        self.weights_dni = 0.2 * np.random.randn(output_dim, output_dim) - 0.1
        self.bias_dni = 0.2 * np.random.randn(output_dim) - 0.1
        self.alpha = alpha

    def forward_and_synthetic_update(self, input, update=True):
        """
        Network forward pass AND update main network weights,
         using Synthetic Gradient
        :return: gradient of the main network, aka "[delta * weights.T]" in original backprop, i.e.
                backpropagate synthetic gradient from the output of the weights to the input,
                so that we can send it to the previous layer;
                forwarded signal
        """
        self.input = input
        # network forward pass
        self.output = self.nonlin(self.input.dot(self.weights) + self.bias)
        if not update:
            return self.output
        else:
            # ***
            # generate synthetic gradient: linear regression
            # ***
            self.syn_grad = self.output.dot(self.weights_dni) + self.bias_dni
            # ***
            # generate synthetic gradient: logistic regression
            # ***
            # self.syn_grad = self.nonlin(self.output.dot(self.weights_dni) + self.bias_dni)
            # update weights of the main network, using synthetic gradient
            delta_weights = self.syn_grad * self.nonlin_deriv(self.output)
            self.weights -= self.alpha * self.input.T.dot(delta_weights)
            self.bias -= self.alpha * np.mean(delta_weights)
            return delta_weights.dot(self.weights.T), self.output

    def forward(self, input):
        """ Calculate forward network activated signal"""
        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights) + self.bias)
        return self.output

    def update_synthetic_weights(self, true_grad):
        """
        backprop for DNI network, true_grad comes from the above layer
        """
        # linear regression generator
        syn_grad_delta = self.syn_grad - true_grad
        # logistic regression generator
        # syn_grad_delta = (self.syn_grad - true_grad) * self.nonlin_deriv(self.syn_grad)
        self.weights_dni -= self.alpha * self.output.T.dot(syn_grad_delta)
        self.bias_dni -= self.alpha * np.mean(syn_grad_delta, axis=0)

    def normal_update(self, true_gradient):
        grad = true_gradient * self.nonlin_deriv(self.output)

        self.weights -= self.alpha * self.input.T.dot(grad)
        self.bias -= self.alpha * np.mean(grad, axis=0)
        return grad.dot(self.weights.T)


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
        epoches = 1000
        X_train, Y_train = data.generate_dataset(num_examples=num_examples, output_dim=output_dim)

        batch_size = 256
        learn_rate = 0.1
        nsamples, input_dim = X_train.shape
        layer_1_dim = 64
        layer_2_dim = 32
        nsamples, output_dim = Y_train.shape
    elif data_type == 'mnist':
        X_train, Y_train, X_val, Y_val, X_test, Y_test = mnist_loader.load_matrices()
        print("MNIST data is loaded...")
        epoches = 500
        batch_size = 10
        learn_rate = .00001  # 3.0
        nsamples, input_dim = X_train.shape
        layer_1_dim = 32
        layer_2_dim = 16
        nsamples, output_dim = Y_train.shape

    # define 3 layers network with synthetic gradients weights
    layer_1 = DNI(input_dim, layer_1_dim, nonl.sigmoid, nonl.sigmoid_derivative, learn_rate)
    layer_2 = DNI(layer_1_dim, layer_2_dim, nonl.sigmoid, nonl.sigmoid_derivative, learn_rate)
    layer_3 = DNI(layer_2_dim, output_dim, nonl.sigmoid, nonl.sigmoid_derivative, learn_rate)

    total_acc, total_cost_1, total_cost_2 = [], [], []

    for iter in range(epoches + 1):
        real_error = 0
        syn_error_2 = 0
        syn_error_1 = 0

        for batch_i in range(int(nsamples / batch_size)):
            batch_x = X_train[(batch_i * batch_size):(batch_i + 1) * batch_size]
            batch_y = Y_train[(batch_i * batch_size):(batch_i + 1) * batch_size]
            # # forward pass
            # _, layer_1_out = layer_1.forward_and_synthetic_update(batch_x)
            # delta_1, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
            # delta_2, layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out)
            # # backward pass
            # # TODO use 2 SG generators
            # delta_3 = layer_3_out - batch_y
            # layer_3.update_synthetic_weights(delta_3)
            # layer_2.update_synthetic_weights(delta_2)
            # layer_1.update_synthetic_weights(delta_1)

            _, layer_1_out = layer_1.forward_and_synthetic_update(batch_x)
            delta_dni_1, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
            layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out, False)

            delta_3 = layer_3_out - batch_y
            delta_2 = layer_3.normal_update(delta_3)
            layer_2.update_synthetic_weights(delta_2)
            layer_1.update_synthetic_weights(delta_dni_1)

            real_error += np.linalg.norm(delta_3) / batch_size
            syn_error_2 += np.linalg.norm(delta_2 - layer_2.syn_grad) / batch_size
            syn_error_1 += np.linalg.norm(delta_dni_1 - layer_1.syn_grad) / batch_size
        if (iter % 5 == 0):
            print("\rIter:" + str(iter) + " Real_error:" + str(real_error) + " Synthetic_2:" + str(syn_error_2) + " Synthetic_1:" + str(syn_error_1))

            # accuracy calculation on VAL dataset
        layer_1_out = layer_1.forward(X_val)
        layer_2_out = layer_2.forward(layer_1_out)
        layer_3_out = layer_3.forward(layer_2_out)
        acc = accuracy(ref=Y_val, pred=layer_3_out)
        total_acc.append(acc)
        total_cost_1.append(syn_error_1)
        total_cost_2.append(syn_error_2)
        print(total_acc[-1])
    ut_plt.show_curves([total_acc],
                       legend=['val error acc'],
                       labels=["# of epochs", "value, %"],
                       title='Error accuracy, sigmoid scores')
    ut_plt.show_curves([total_cost_1, total_cost_2],
                       legend=['syn_error_1', 'syn_error_2'],
                       labels=["# of epochs", "error value"],
                       title='Error, sigmoid scores')
