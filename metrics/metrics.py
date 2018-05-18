import numpy as np
import sklearn.metrics as sk_metrics
from sklearn.isotonic import IsotonicRegression


def eer(y_true, y_score):
    """
    y_true: array of ground truth
    y_score: corresponding scores
    """
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=True)

    eps = 1E-6
    Points = [(0, 0)] + zip(fpr, tpr)
    for i, point in enumerate(Points):
        if point[0] + eps >= 1 - point[1]:
            break
    P1 = Points[i - 1]
    P2 = Points[i]
    # Interpolate between P1 and P2
    if abs(P2[0] - P1[0]) < eps:
        EER = P1[0]
    else:
        m = (P2[1] - P1[1]) / (P2[0] - P1[0])
        o = P1[1] - m * P1[0]
        EER = (1 - o) / (1 + m)
    return EER


def sklearn_rocch(y_true, y_score):
    """
    Binary ROC convex hull.
    NOTE: sklearn isotonic regression is used
    y_true: 1D array
    y_score: 1D array
    """
    y_calibr, p_calibr = sklearn_pav(y_true=y_true, y_score=y_score)
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_calibr, p_calibr, drop_intermediate=True)
    return fpr, tpr, thresholds, y_calibr, p_calibr


def sklearn_pav(y_true, y_score):
    """
    Binary PAV algorithm, algorithm to solve Isotonic regression
    NOTE: sklearn isotonic regression is used
    y_true: 1D array
    y_score: 1D array
    """
    id_permute = np.argsort(y_score)
    y_sort = y_true[id_permute]
    p_sort = np.sort(y_score)

    ir = IsotonicRegression()
    p_calibrated = ir.fit_transform(p_sort, y_sort)
    return y_sort, p_calibrated


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


def discrete_error_rates(y_true, y_score):
    """
    :return: lists: tpr, tnr, fpr, fnr, thresholds
    """
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=True)
    fpr = np.insert(fpr, 0, 0.)
    tpr = np.insert(tpr, 0, 0.)
    fnr = 1. - tpr
    tnr = 1. - fpr
    thresholds = np.insert(thresholds, 0, 1.)
    return tpr, tnr, fpr, fnr, thresholds
