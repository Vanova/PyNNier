import numpy as np
import pandas as pd

# test scores
p_test = np.array([[0.6, 0.8, 0.7, 0.9],
                   [0.7, 0.2, 0.3, 0.4],
                   [0.1, 0.4, 0.3, 0.0],
                   [0.6, 0.2, 0.1, 0.3],
                   [0.7, 0.8, 0.6, 0.9],
                   [0.0, 0.3, 0.7, 0.4],
                   [0.1, 0.8, 0.9, 0.3],
                   [0.2, 0.6, 0.1, 0.3],
                   [0.6, 0.9, 0.6, 0.3],
                   [0.1, 0.3, 0.4, 0.2]])
# ground-truth
y_test = np.array([[1, 1, 1, 1],
                   [0, 0, 0, 1],
                   [1, 0, 0, 0],
                   [1, 1, 1, 1],
                   [0, 0, 1, 1],
                   [0, 1, 1, 0],
                   [0, 1, 1, 1],
                   [1, 0, 0, 1],
                   [1, 1, 1, 1],
                   [0, 0, 1, 0]])

# p = [['f1', 'a', 0.7], ['f1', 'b', 0.4], ['f1', 'c', 0.9], ['f2', 'a', 0.7], ['f2', 'b', 0.4], ['f2', 'c', 0.9],
#      ['f3', 'a', 0.1], ['f3', 'b', 0.5], ['f3', 'c', 0.3]]
# p_df = pd.DataFrame(p)
# p_df.pivot(index=0, columns=1, values=2)

# y = [['f2', 'a', 0.7], ['f2', 'c', 0.4], ['f2', 'b', 0.9], ['f1', 'a', 0.7], ['f1', 'b', 0.4], ['f1', 'c', 0.9],
#      ['f3', 'a', 0.1], ['f3', 'b', 0.5], ['f3', 'c', 0.3]]
# y_df = pd.DataFrame(y)
# y_df.pivot(index=0, columns=1, values=2)

def arr2DataFrame(arr, row_id=None, col_id=None):
    """
    Transform numpy 2D array to panda DataFrame
    """
    row, col = arr.shape
    idx = pd.Index(range(row))
    df = pd.DataFrame(arr, index=idx, columns=['C_%d' % i for i in range(col)])
    return df


def class_wise_tnt(p, y):
    """
    Split scores on target and non-target, across the classes (columns)
    p: predicted scores, DataFrame
    y: target labels, DataFrame
    :return: two lists of DataFrames, target and non-target scores
    """
    ts = []
    nts = []
    for c in y.columns:
        t = pd.DataFrame({c + '_1': p[c][y[c] > 0].values},
                           columns=[c + '_1'])
        nt = pd.DataFrame({c + '_0': p[c][y[c] < 1].values}, columns=[c + '_0'])
        ts.append(t)
        nts.append(nt)
    return ts, nts


def pooled_tnt(p, y):
    """
    Split only targets and only non-target scores across all classes
    :return: target and non-target 1D arrays
    """
    ts_pool = []
    nts_pool = []
    for c in y.columns:
        ts_pool.extend(p[c][y[c] > 0].values)
        nts_pool.extend(p[c][y[c] < 1].values)
    return np.array(ts_pool), np.array(nts_pool)


def pooled_scores(p, y):
    """
    Pool all target and non-target scores across all classes
    :return: column of scores and target labels, DataFrame
    """
    pst = p.stack()
    yst = y.stack()
    pool = pd.concat([pst, yst], axis=1)
    return pool
