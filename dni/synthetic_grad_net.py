"""
Synthetic Gradient network, summation of two binary digits
"""
import numpy as np
import data
import ann.lazy.nonlinearity as nonl


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

    def forward_and_synthetic_update(self, input):
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

        # ***
        # generate synthetic gradient: linear regression
        # ***
        self.syn_grad = self.nonlin(self.output.dot(self.weights_dni) + self.bias_dni)

        # update weights of the main network, using synthetic gradient
        delta_weights = self.syn_grad * self.nonlin_deriv(self.output)
        # TODO test +
        self.weights -= self.alpha * self.input.T.dot(delta_weights)
        self.bias -= self.alpha * np.mean(delta_weights)
        return delta_weights.dot(self.weights.T), self.output

    def update_synthetic_weights(self, true_grad):
        """
        backprop for DNI network, true_grad comes from the above layer
        """
        # TODO test +
        syn_grad_delta = (self.syn_grad - true_grad) * self.nonlin_deriv(self.syn_grad)
        self.weights_dni -= self.alpha * self.output.T.dot(syn_grad_delta)
        self.bias_dni -= self.alpha * np.mean(syn_grad_delta, axis=0)


if __name__ == '__main__':
    from utils import mnist_loader

    # TODO add metric calculation
    data_type = 'mnist'

    if data_type == 'toy':
        num_examples = 1000
        output_dim = 12
        epoches = 1000
        X, Y = data.generate_dataset(num_examples=num_examples, output_dim=output_dim)

        batch_size = 256
        learn_rate = 0.1
        nsamples, input_dim = X.shape
        layer_1_dim = 64
        layer_2_dim = 32
        nsamples, output_dim = Y.shape
    elif data_type == 'mnist':
        X_train, Y_train, X_val, Y_val, X_test, Y_test = mnist_loader.load_matrices()
        print("MNIST data is loaded...")
        # MSE network with sigmoid output layer
        epoches = 30
        batch_size = 10
        learn_rate = 3. # 3.0
        nsamples, input_dim = X_train.shape
        layer_1_dim = 64
        layer_2_dim = 32
        nsamples, output_dim = Y_train.shape

    # define 3 layers network with synthetic gradients weights
    layer_1 = DNI(input_dim, layer_1_dim, nonl.sigmoid, nonl.sigmoid_derivative, learn_rate)
    layer_2 = DNI(layer_1_dim, layer_2_dim, nonl.sigmoid, nonl.sigmoid_derivative, learn_rate)
    layer_3 = DNI(layer_2_dim, output_dim, nonl.sigmoid, nonl.sigmoid_derivative, learn_rate)

    for iter in range(epoches+1):
        error = 0
        synthetic_error = 0

        for batch_i in range(int(nsamples / batch_size)):
            batch_x = X_train[(batch_i * batch_size):(batch_i + 1) * batch_size]
            batch_y = Y_train[(batch_i * batch_size):(batch_i + 1) * batch_size]

            # forward pass
            _, layer_1_out = layer_1.forward_and_synthetic_update(batch_x)
            delta_1, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
            delta_2, layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out)
            # backward pass
            delta_3 = layer_3_out - batch_y
            layer_3.update_synthetic_weights(delta_3)
            layer_2.update_synthetic_weights(delta_2)
            layer_1.update_synthetic_weights(delta_1)

            # error += (np.sum(np.abs(delta_3 * layer_3_out * (1 - layer_3_out))))
            error += np.sum(np.abs(delta_3 * nonl.sigmoid_derivative(layer_3_out)))
            synthetic_error += (np.sum(np.abs(delta_2 - layer_2.syn_grad)))
        if (iter % 5 == 0):
            print("\rIter:" + str(iter) + " Loss:" + str(error) + " Synthetic Loss:" + str(synthetic_error))
