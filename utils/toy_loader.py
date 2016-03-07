import numpy as np
from sklearn.datasets import make_multilabel_classification

RANDOM_SEED = 888


def load_data(n_tr=250, n_dev=50, n_tst=50, n_features=2):
    n_samples = n_tr + n_tst + n_dev
    # generate data
    x, y, p_c, p_w_c = make_multilabel_classification(n_samples=n_samples, n_features=n_features,
                                                      n_classes=2, n_labels=1,
                                                      allow_unlabeled=False,
                                                      return_distributions=True,
                                                      random_state=RANDOM_SEED)
    data = nnet_format(zip(x, y))
    # split data
    tr = data[:n_tr]
    tst = data[n_tr:n_tr + n_tst]
    dev = data[n_tr + n_tst:]
    return (tr, dev, tst)


def nnet_format(data):
    """Input: list of tuples (x, y), x and y arrays.
    Return list of tuples with columns arrays"""
    xn = data[0][0].size
    yn = data[0][1].size
    return [(x.reshape((xn, 1)), y.reshape((yn, 1))) for x, y in data]


def plot_format(data):
    """Return lists xs, ys with row arrays"""
    xs = []
    ys = []
    for x, y in data:
        it = x.T.tolist()
        xs.append(it[0])
        it = y.T.tolist()
        ys.append(it[0])
    return (np.array(xs), np.array(ys))
