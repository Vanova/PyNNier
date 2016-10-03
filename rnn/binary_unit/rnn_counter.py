# Vanilla RNN for counting units in binary number
import numpy as np
# import rnn_plotter as rp
# import matplotlib.pyplot as plt
from utils.plotters import NetworkVisualiser

np.random.seed(seed=1)


def dataset(num_samples=20, seq_len=10):
    # Create the sequences
    x = np.zeros((nb_of_samples, sequence_len))
    for row_idx in range(nb_of_samples):
        x[row_idx, :] = np.around(np.random.rand(sequence_len)).astype(int)
    # Create the targets for each sequence
    t = np.sum(x, axis=1)
    return (x, t)


class RNNBinaryCounter():
    def __init__(self):
        self.W = [-1.5, 2.]  # wx, wRec

    # Define the forward step functions
    def update_state(self, xk, sk, wx, wRec):
        """
        Compute state k from the previous state (sk) and current input (xk),
        by use of the input weights (wx) and recursive weights (wRec).
        """
        return xk * wx + sk * wRec

    def forward_states(self, X, wx, wRec):
        """
        Unfold the network and compute all state activations given the input X,
        and input weights (wx) and recursive weights (wRec).
        Return the state activations in a matrix, the last column S[:,-1] contains the
        final activations.
        """
        # Initialise the matrix that holds all states for all input sequences.
        # The initial state s0 is set to 0.
        S = np.zeros((X.shape[0], X.shape[1] + 1))
        for k in range(0, X.shape[1]):
            S[:, k + 1] = self.update_state(X[:, k], S[:, k], wx, wRec)
        return S

    def backward_gradient(self, X, S, grad_out, wRec):
        """
        Backpropagate the gradient computed at the output (grad_out) through the network.
        Accumulate the parameter gradients for wX and wRec by for each layer by addition.
        Return the parameter gradients as a tuple, and the gradients at the output of each layer.
        """
        # Initialise the array that stores the gradients of the cost with respect to the states.
        grad_over_time = np.zeros((X.shape[0], X.shape[1] + 1))
        grad_over_time[:, -1] = grad_out
        # Set the gradient accumulations to 0
        wx_grad = 0
        wRec_grad = 0
        for k in range(X.shape[1], 0, -1):
            # Compute the parameter gradients and accumulate the results.
            wx_grad += np.sum(grad_over_time[:, k] * X[:, k - 1])
            wRec_grad += np.sum(grad_over_time[:, k] * S[:, k - 1])
            # Compute the gradient at the output of the previous layer
            grad_over_time[:, k - 1] = grad_over_time[:, k] * wRec
        return (wx_grad, wRec_grad), grad_over_time

    def output_gradient(self, y, t):
        """
        Compute the gradient of the MSE cost function with respect to the output y.
        """
        n_samples = t.shape[0]
        return 2.0 * (y - t) / n_samples

    def get_grad_over_time(self, wx, wRec, data, lab):
        """Helper func to only get the gradient over time from wx and wRec."""
        S = self.forward_states(data, wx, wRec)
        grad_out = self.output_gradient(S[:, -1], lab).sum()
        _, grad_over_time = self.backward_gradient(data, S, grad_out, wRec)
        return grad_over_time

    def cost_value(self, y, t):
        """
        Return the MSE between the targets t and the outputs y.
        """
        n_samples = t.shape[0]
        return ((t - y) ** 2).sum() / n_samples

    def rprop_training(self, eta_p, eta_n):
        W_delta = [0.001, 0.001]  # Update values (Delta) for W
        W_sign = [0, 0]  # Previous sign of W
        list_ws = [(self.W[0], self.W[1])]  # List of weights to plot

        for i in range(500):
            # Get the update values and sign of the last gradient
            W_delta, W_sign = network.step_rprop(data, labels, W_sign, W_delta, eta_p, eta_n)
            # Update each weight parameter separately
            for i, _ in enumerate(self.W):
                self.W[i] -= W_sign[i] * W_delta[i]
            list_ws.append((self.W[0], self.W[1]))  # Add weights to list to plot

        print('Final weights are: wx = {0},  wRec = {1}'.format(self.W[0], self.W[1]))

        return list_ws

    # Define Rprop optimisation function
    def step_rprop(self, x, t, W_prev_sign, W_delta, eta_p, eta_n):
        """
        Update Rprop values in one iteration.
        X: input data.
        t: targets.
        W_prev_sign: Previous sign of the W gradient.
        W_delta: Rprop update values (Delta).
        eta_p, eta_n: Rprop hyperparameters.
        """
        # Perform forward and backward pass to get the gradients
        S = self.forward_states(x, self.W[0], self.W[1])
        grad_out = self.output_gradient(S[:, -1], t)
        W_grads, _ = self.backward_gradient(x, S, grad_out, self.W[1])
        W_sign = np.sign(W_grads)  # Sign of new gradient
        # Update the Delta (update value) for each weight parameter seperately
        for i, _ in enumerate(self.W):
            if W_sign[i] == W_prev_sign[i]:
                W_delta[i] *= eta_p
            else:
                W_delta[i] *= eta_n
        return W_delta, W_sign


def gradient_check(X, Y, net):
    """
    Numerical check of the gradient with the backprop gradient
    """
    # Perform gradient checking
    # Set the weight parameters used during gradient checking
    params = [1.2, 1.2]  # [wx, wRec]
    # Set the small change to compute the numerical gradient
    eps = 1e-7
    # Compute the backprop gradients
    S = net.forward_states(X, params[0], params[1])
    grad_out = net.output_gradient(S[:, -1], Y)
    backprop_grads, grad_over_time = net.backward_gradient(X, S, grad_out, params[1])

    # Compute the numerical gradient for each parameter in the layer
    for p_idx, _ in enumerate(params):
        grad_backprop = backprop_grads[p_idx]
        # + eps
        params[p_idx] += eps
        plus_cost = net.cost_value(net.forward_states(X, params[0], params[1])[:, -1], Y)
        # - eps
        params[p_idx] -= 2 * eps
        minus_cost = net.cost_value(net.forward_states(X, params[0], params[1])[:, -1], Y)
        # reset param value
        params[p_idx] += eps
        # calculate numerical gradient
        grad_num = (plus_cost - minus_cost) / (2 * eps)
        # Raise error if the numerical grade is not close to the backprop gradient
        if not np.isclose(grad_num, grad_backprop):
            raise ValueError(
                'Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(
                    float(grad_num), float(grad_backprop)))
    print('No gradient errors found')



# create dataset
nb_of_samples = 20
sequence_len = 10
data, labels = dataset(num_samples=nb_of_samples, seq_len=sequence_len)

# # Plot cost surface and gradients
# # Get and plot the cost surface figure with markers
# fig = rp.get_cost_surface_figure(lambda w1, w2: cost(forward_states(data, w1, w2)[:, -1], labels), rp.points)
# # Get the plots of the gradients changing by backpropagation.
# rp.plot_gradient_over_time(rp.points, get_grad_over_time, data, labels)
# # Show figures
# plt.show()

network = RNNBinaryCounter()
gradient_check(data,labels, network)
# Rprop training
eta_p = 1.2
eta_n = 0.5
list_of_ws = network.rprop_training(eta_p, eta_n)
# plot the cost surface with the weights over the iterations
net_viz = NetworkVisualiser()
net_viz.plot_optimisation(list_of_ws, lambda w1, w2: network.cost_value(network.forward_states(data, w1, w2)[:, -1], labels))
