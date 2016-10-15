import numpy as np
from sklearn.datasets import make_multilabel_classification

RANDOM_SEED = 888


def load_data(n_tr=250, n_dev=50, n_tst=50, n_features=2, n_classes=2):
    n_samples = n_tr + n_tst + n_dev
    # generate data
    x, y, p_c, p_w_c = make_multilabel_classification(n_samples=n_samples, n_features=n_features,
                                                      n_classes=n_classes, n_labels=1,
                                                      allow_unlabeled=False,
                                                      return_distributions=True,
                                                      random_state=RANDOM_SEED)
    data = nnet_format(x, y)
    # split data
    tr = data[:n_tr]
    tst = data[n_tr:n_tr + n_tst]
    dev = data[n_tr + n_tst:]
    return tr, dev, tst


def nnet_format(x, y):
    """Input: 2D arrays x and y.
    Return list of tuples with columns arrays [dim, 1]"""
    return [(x.reshape((-1, 1)), y.reshape((-1, 1))) for x, y in zip(x, y)]


def plot_format(data):
    """
    Input: list of tuples of arrays
    Return 2D arrays xs, ys """
    xs = []
    ys = []
    for x, y in data:
        xs.append(x.flatten())
        ys.append(y.flatten())
    return np.array(xs), np.array(ys)


def plot_format_no_ticks(data):
    """Return lists xs, ys with row arrays"""
    xs = []
    for x in data:
        xs.append(x.flatten())
    return np.array(xs)
