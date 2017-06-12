import numpy as np
import pandas as pd

import metrics as mfom_metr

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
y_test = np.array([[0, 1, 1, 1],
                   [1, 0, 0, 1],
                   [0, 0, 1, 0],
                   [1, 1, 0, 1],
                   [0, 0, 1, 1],
                   [0, 1, 1, 0],
                   [0, 1, 1, 1],
                   [0, 0, 0, 1],
                   [1, 1, 0, 1],
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
    if len(arr.shape) > 1:
        row, col = arr.shape
    else:
        col = 1
        row = len(arr)

    if row_id is None:
        row_id = pd.Index(range(row))

    if col_id is None:
        col_id = ['C_%d' % i for i in range(col)]

    df = pd.DataFrame(arr, index=row_id, columns=col_id)
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


# def array_tnt(p, y):
#     """
#     Return target part of scores and non-target
#     p: 1D array, predicted scores
#     y: 1D array, ground truth
#     """
#     return p[y > 0], p[y < 1]


def pool_split_tnt(p_df, y_df):
    """
    Split only targets and only non-target scores across all classes
    p: DataFrame
    y: DataFrame
    :return: target 1D array, non-target 1D array
    """
    # TODO refactor
    ts_pool = []
    nts_pool = []
    for c in y_df.columns:
        ts_pool.extend(p_df[c][y_df[c] > 0].values)
        nts_pool.extend(p_df[c][y_df[c] < 1].values)
    return np.array(ts_pool), np.array(nts_pool)


# def dataframe_split_tnt(p_df, y_df):
#     """
#     Split only targets and only non-target scores across all classes
#     p: DataFrame
#     y: DataFrame
#     :return: target 1D array, non-target 1D array
#     """
#     # TODO refactor
#     ts_pool = []
#     nts_pool = []
#     for c in y_df:
#         ts_pool.extend(p_df[c][y_df[c] > 0].values)
#         nts_pool.extend(p_df[c][y_df[c] < 1].values)
#     return np.array(ts_pool), np.array(nts_pool)


def pool_scores(p_df, y_df):
    """
    Pool all target and non-target scores across all classes
    :return: pooled scores DataFrame, pooled target DataFrame
    """
    pst = p_df.stack()
    yst = y_df.stack()
    pool = pd.concat([pst, yst], axis=1)
    return pool[0], pool[1]


def calibrate_scores(p_df, y_df):
    """
    p_df: DataFrame, [samples x classes]
    y_df: DataFrame, [samples x classes]
    :return: calibrated Y and P DataFrames
    """
    Y = []
    P = []
    for yc in y_df:
        # calibrated scores
        y_true = y_df[yc].values
        y_score = p_df[yc].values
        y_calibr, p_calibr = mfom_metr.sklearn_pav(y_true=y_true, y_score=y_score)
        Y.append(y_calibr)
        P.append(p_calibr)
    # wrap with DataFrame
    Y = np.asarray(Y).T
    P = np.asarray(P).T
    return arr2DataFrame(Y, col_id=y_df.columns), \
           arr2DataFrame(P, col_id=y_df.columns)


