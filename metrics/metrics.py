import numpy as np


def micro_f1(y_true, y_pred, accuracy=True):
    """
    :param y_true, y_pred: binary integer list of ndarrays,
        several hot-labels
    :return: accuracy or error of micro F1
    """
    # TODO example of usage
    assert (len(y_true) == len(y_pred))
    neg_r = np.logical_not(y_true)
    neg_p = np.logical_not(y_pred)
    tp = np.sum(np.logical_and(y_true, y_pred) == True)
    fp = np.sum(np.logical_and(neg_r, y_pred) == True)
    fn = np.sum(np.logical_and(y_true, neg_p) == True)
    f1 = 100.0 * 2.0 * tp / (2.0 * tp + fp + fn)
    return accuracy and f1 or 100.0 - f1


def step(a, threshold=0.0):
    """Heaviside step function:
    a < threshold = 0, else 1.
    :param a: array
    :return: array
    """
    res = np.zeros_like(a)
    res[a < threshold] = 0
    res[a >= threshold] = 1
    return res


def pooled_accuracy(y_true, y_pred):
    """
    y_true: 2D array
    y_pred: 2D array
    """
    assert (len(y_true) == len(y_pred))
    p = np.argmax(y_pred, axis=1)
    y = np.argmax(y_true, axis=1)
    N = float(len(y_true))
    return np.sum(p == y) / N * 100.
