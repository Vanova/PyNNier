import numpy as np

def micro_f1(refs, predicts, accuracy=True):
    """

    :param refs, predicts: binary integer list of ndarrays,
        several hot-labels
    :return: accuracy or error of micro F1
    """
    assert(len(refs) == len(predicts))
    neg_r = np.logical_not(refs)
    neg_p = np.logical_not(predicts)
    tp = np.sum(np.logical_and(refs, predicts) == True)
    fp = np.sum(np.logical_and(neg_r, predicts) == True)
    fn = np.sum(np.logical_and(refs, neg_p) == True)
    f1 = 100.0 * 2.0 * tp / (2.0*tp + fp + fn)
    return accuracy and f1 or 100.0 - f1
