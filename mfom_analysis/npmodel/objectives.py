"""
Ref.: see my paper
"""
_EPSILON = 1e-7

# def backend(b):
#     if b == 'numpy':
#         import numpy as np
#     elif b == 'autograd':
#         import autograd.numpy as np


def mfom_eer_uvz(y_true, y_pred, alpha=3., beta=0.):
    """
    MFoM-EER with 'Units-vs-zeros' strategy, matrix numpy implementation.
    Optimized implementation of choosing anti-model, either belonging to zero
    or unit sets.
    y_true: [batch_sz, nclasses]
    y_pred: sigmoid scores, we preprocess these to d_k and l_k,
    i.e. loss function l_k(Z)
    """
    y_neg = 1 - y_true  # or y = np.logical_not(y_true)
    # number of negative samples per each class
    N = np.sum(y_neg, axis=0, keepdims=True)
    # number of positive samples per each class
    P = np.sum(y_true, axis=0, keepdims=True)
    # calculate class loss function l
    l = _uvz_loss_scores(y_true, y_pred, alpha, beta, is_training=True)

    # ===
    # smoothed EER
    # ===
    fn = l * y_true
    fp = (1. - l) * y_neg
    # sum across samples
    fnr = np.sum(fn, axis=0, keepdims=True) / P
    fpr = np.sum(fp, axis=0, keepdims=True) / N
    # smooth_eer = fn + 0.1 * np.abs(fn - fp) # regularized
    # simplified smooth EER
    smooth_eer = np.abs(fnr + fpr)
    return fnr, fpr, np.mean(smooth_eer)


def _uvz_loss_scores(y_true, y_pred, alpha=1., beta=0., is_training=True):
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)

    if is_training:
        y_pred = np.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
        y_neg = 1 - y_true  # or y = np.logical_not(y_true)
        # Kolmogorov log average of unit labeled models
        unit_avg = y_true * np.exp(y_pred)
        # average over non-zero elements
        unit_avg = np.log(_non_zero_mean_np(unit_avg))
        # Kolmogorov log average of zero labeled models
        zeros_avg = y_neg * np.exp(y_pred)
        # average over non-zero elements
        zeros_avg = np.log(_non_zero_mean_np(zeros_avg))
        # misclassification measure, optimized
        d = -y_pred + y_neg * unit_avg + y_true * zeros_avg
    else:
        d = -y_pred + 0.5

    # calculate class loss function l
    l = 1.0 / (1.0 + np.exp(-alpha * d - beta))
    return l


def _ovo_loss_scores(y_true, y_pred, alpha=3., beta=0.):
    """
    Ref: see equation ... from my paper
    """
    y_pred = np.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    nclass = y_true.shape[-1]
    # softmax normalization: ssigma in report
    norms = softmax(y_pred)
    # misclassification measure
    d = np.log(1. / (nclass - 1) * (1. / norms - 1.) + _EPSILON)
    # calculate class loss function l
    l = 1.0 / (1.0 + np.exp(-alpha * d - beta))
    return l


def _non_zero_mean_np(x):
    # Average values which meet the criterion > 0
    mask = (np.abs(x) > _EPSILON)
    n = np.sum(mask, axis=-1, keepdims=True)
    return np.sum(x, axis=-1, keepdims=True) / n
    # return np.true_divide(x.sum(axis=-1, keepdims=True), (np.abs(x) > 0).sum(axis=-1, keepdims=True))


def softmax(x):
    """Compute softmax values for each sets of scores in x.
    x: [samples; dim]
    """
    xt = x.T
    e_x = np.exp(xt - np.max(xt))
    s = e_x / e_x.sum(axis=0)
    return s.T


def sigma(x):
    return 1. / (1. + np.exp(-x))