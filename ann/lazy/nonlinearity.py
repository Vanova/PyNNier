import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x + 1e-6))


def sigmoid_linear(z, a=1., b=0.):
    return 1.0 / (1.0 + np.exp(-a * z + b))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid_derivative(x):
    """Derivative of the sigmoid function wrt network output"""
    return x * (1 - x)


def softmax(z):
    """
    Normalization across the row vector
    :param z: 2D array, smp x dim
    """
    zt = z.transpose()
    x = zt - np.max(zt, axis=0)  # safe explosion trick
    p = np.exp(x) / np.sum(np.exp(x), axis=0)
    return p.transpose()
