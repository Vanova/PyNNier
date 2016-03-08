import numpy as np

def micro_f1(refs, predicts, accuracy=True):
    """Input: binary integer list of ndarrays,
    several hot-labels"""
    assert(len(refs) == len(predicts))
    neg_r = np.logical_not(refs)
    neg_p = np.logical_not(predicts)
    tp = np.sum(np.logical_and(refs, predicts) == True)
    fp = np.sum(np.logical_and(neg_r, predicts) == True)
    fn = np.sum(np.logical_and(refs, neg_p) == True)
    f1 = 100.0 * 2.0 * tp / (2.0*tp + fp + fn)
    return accuracy and f1 or 100.0 - f1


def step(a, threshold=0.0):
    """Heaviside step function:
    a < threshold = 0, else 1.
    Input: array
    Output: array"""
    res = [[0] if x < threshold else [1] for x in a ]
    return np.array(res) # need column
