# Vanilla RNN for counting units in binary number
import numpy as np
import rnn_plotter as rp
import matplotlib.pyplot as plt

np.random.seed(seed=1)


def dataset(num_samples=20, seq_len=10):
    # Create the sequences
    X = np.zeros((nb_of_samples, sequence_len))
    for row_idx in range(nb_of_samples):
        X[row_idx, :] = np.around(np.random.rand(sequence_len)).astype(int)
    # Create the targets for each sequence
    t = np.sum(X, axis=1)
    return (X, t)


# Define the forward step functions
def update_state(xk, sk, wx, wRec):
    """
    Compute state k from the previous state (sk) and current input (xk),
    by use of the input weights (wx) and recursive weights (wRec).
    """
    return xk * wx + sk * wRec


def forward_states(X, wx, wRec):
    """
    Unfold the network and compute all state activations given the input X,
    and input weights (wx) and recursive weights (wRec).
    Return the state activations in a matrix, the last column S[:,-1] contains the
    final activations.
    """
    # Initialise the matrix that holds all states for all input sequences.
    # The initial state s0 is set to 0.
    S = np.zeros((X.shape[0], X.shape[1] + 1))
    # Use the recurrence relation defined by update_state to update the
    #  states trough time.
    for k in range(0, X.shape[1]):
        # S[k] = S[k-1] * wRec + X[k] * wx
        S[:, k + 1] = update_state(X[:, k], S[:, k], wx, wRec)
    return S


def backward_gradient(X, S, grad_out, wRec):
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


def cost(y, t):
    """
    Return the MSE between the targets t and the outputs y.
    """
    n_samples = t.shape[0]
    return ((t - y) ** 2).sum() / n_samples


def output_gradient(y, t):
    """
    Compute the gradient of the MSE cost function with respect to the output y.
    """
    n_samples = t.shape[0]
    return 2.0 * (y - t) / n_samples


def gradient_check(X, Y):
    """
    Numerical check of the gradient with the backprop gradient
    """
    # Perform gradient checking
    # Set the weight parameters used during gradient checking
    params = [1.2, 1.2]  # [wx, wRec]
    # Set the small change to compute the numerical gradient
    eps = 1e-7
    # Compute the backprop gradients
    S = forward_states(X, params[0], params[1])
    grad_out = output_gradient(S[:, -1], Y)
    backprop_grads, grad_over_time = backward_gradient(X, S, grad_out, params[1])

    # Compute the numerical gradient for each parameter in the layer
    for p_idx, _ in enumerate(params):
        grad_backprop = backprop_grads[p_idx]
        # + eps
        params[p_idx] += eps
        plus_cost = cost(forward_states(X, params[0], params[1])[:, -1], Y)
        # - eps
        params[p_idx] -= 2 * eps
        minus_cost = cost(forward_states(X, params[0], params[1])[:, -1], Y)
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


def get_grad_over_time(wx, wRec, data, lab):
    """Helper func to only get the gradient over time from wx and wRec."""
    S = forward_states(data, wx, wRec)
    grad_out = output_gradient(S[:, -1], lab).sum()
    _, grad_over_time = backward_gradient(data, S, grad_out, wRec)
    return grad_over_time


# create dataset
nb_of_samples = 20
sequence_len = 10
data, labels = dataset(num_samples=nb_of_samples, seq_len=sequence_len)
# check the numerical gradient
gradient_check(X=data, Y=labels)

# Plot cost surface and gradients
# Get and plot the cost surface figure with markers
fig = rp.get_cost_surface_figure(lambda w1, w2: cost(forward_states(data, w1, w2)[:, -1], labels), rp.points)
# Get the plots of the gradients changing by backpropagating.
rp.plot_gradient_over_time(rp.points, get_grad_over_time, data, labels)
# Show figures
plt.show()
