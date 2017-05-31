"""
Compare discrete and convex hull ROC computation
and EER from these plots
"""
import numpy as np
import sklearn.metrics as sk_metrics
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from mfom.utils import toy_scores as TS
import mfom.utils.plotter as mfom_plt
from metrics import eer, sklearn_rocch
from matplotlib.pyplot import cm


def toy_score_table(p_df, y_df):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 1, 1, xticks=[], yticks=[], frameon=False)
    mfom_plt.colored_table(ax, vals=p_df.values, col_lab=p_df.columns, row_lab=p_df.index)

    ax = fig.add_subplot(2, 1, 2, xticks=[], yticks=[], frameon=False)
    mfom_plt.colored_table(ax, vals=y_df.values, col_lab=y_df.columns, row_lab=y_df.index)
    plt.show()


def plot_histogram(tar, ntar, bins=5):
    """
    tar: 1D array
    ntar: 1D array
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, frameon=True)

    mfom_plt.view_histogram(ax, tar, ntar, bins)
    fig.tight_layout()
    plt.show()


def plot_roc(y_true, y_score):
    """
    y_true: 1D array
    y_score: 1D array
    """
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=True)
    roc_auc = sk_metrics.auc(fpr, tpr)
    eer_val = eer(y_true=y_true, y_score=y_score)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    mfom_plt.view_roc_curve(ax, fpr, tpr, roc_auc=roc_auc, eer_val=eer_val)
    fig.tight_layout()
    plt.show()


def plot_roc_fnr_fpr(y_true, y_score):
    """
    y_true: 1D array
    y_score: 1D array
    """
    # calculate ROC
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=True)
    fpr = np.insert(fpr, 0, 0.)
    tpr = np.insert(tpr, 0, 0.)
    fnr = 1. - tpr
    thresholds = np.insert(thresholds, 0, 1.)
    eer_val = eer(y_true, y_score)

    # plot FNR/FPR distributions
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    mfom_plt.view_fnr_fpr_dist(ax, fnr, fpr, thresholds, eer_val)
    fig.tight_layout()
    plt.show()


def plot_rocch(y_true, y_score):
    """
        y_true: 1D array
        y_score: 1D array
        """
    # fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=True)
    # roc_auc = sk_metrics.auc(fpr, tpr)
    # eer_val = eer(y_true=y_true, y_score=y_score)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, frameon=True)
    # mfom_plt.view_roc_curve(ax, fpr, tpr, roc_auc=roc_auc, eer_val=eer_val)
    # fig.tight_layout()
    # plt.show()


    fpr, tpr = sklearn_rocch(y_true, y_score)
    roc_auc = sk_metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, marker='o', linestyle='--', color='darkorange', label='ROCCH curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC convex hull')
    plt.legend(loc="lower right")
    plt.show()


def class_wise_scatter(data_frame):
    scatter_matrix(data_frame, alpha=0.5, figsize=(6, 6), diagonal='kde')
    plt.show()


def class_wise_histograms(tars, ntars, bins=10):
    """
    tars: list of DataFrames
    ntars: list of DataFrames
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    n_row = 2
    n_col = np.round(len(tars) / float(n_row))
    i = 1
    for t, nt in zip(tars, ntars):
        ax = fig.add_subplot(n_row, n_col, i, frameon=True)
        mfom_plt.view_histogram(ax, t.values, nt.values, bins)
        i += 1
    fig.tight_layout()
    plt.show()


def class_wise_roc(y_true_cw, y_score_cw):
    """
    y_true_cw: DataFrame, [samples x classes]
    y_score_cw: DataFrame, [samples x classes]
    """
    # calculate fpr/tpr per class
    fprs, tprs, aucs, eer_vals = [], [], [], []
    n_classes = len(y_true_cw.columns)
    for c in range(n_classes):
        y_true = y_true_cw.values[:, c]
        y_score = y_score_cw.values[:, c]
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=True)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(sk_metrics.auc(fpr, tpr))
        eer_vals.append(eer(y_true, y_score))

    # plot curves
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    color = iter(cm.rainbow(np.linspace(0, 1, n_classes)))
    for c in range(n_classes):
        mfom_plt.view_roc_curve(ax, fprs[c], tprs[c], roc_auc=aucs[c], eer_val=eer_vals[c], color=next(color))
    fig.tight_layout()
    plt.show()


def class_wise_roc_fnr_fpr(y_true_cw, y_score_cw):
    """
    y_true_cw: DataFrame, [samples x classes]
    y_score_cw: DataFrame, [samples x classes]
    """
    # calculate fpr/tpr per class
    fprs, fnrs, thresholds, eer_vals = [], [], [], []
    for cname in y_true_cw:
        y_true = y_true_cw[cname].values
        y_score = y_score_cw[cname].values
        fpr, tpr, thresh = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=True)
        # TODO: fix plot values
        # thresh = np.insert(thresh, 0, 1.)
        # fpr = np.insert(fpr, 0, 0.)
        # tpr = np.insert(tpr, 0, 0.)
        fnr = 1. - tpr
        # append for each class
        fprs.append(fpr)
        fnrs.append(fnr)
        thresholds.append(thresh)
        eer_vals.append(eer(y_true, y_score))

    # plot FNR/FPR distributions
    fig = plt.figure(figsize=plt.figaspect(0.5))
    n_row = 2
    n_col = np.round(len(y_true_cw.columns) / float(n_row))
    i = 1
    for fnr, fpr, th, er in zip(fnrs, fprs, thresholds, eer_vals):
        ax = fig.add_subplot(n_row, n_col, i, frameon=True)
        mfom_plt.view_fnr_fpr_dist(ax, fnr, fpr, th, er)
        i += 1
    fig.tight_layout()
    plt.show()


def class_wise_rocch(y_true_df, y_score_df):
    """
    y_true: DataFrame, [samples x classes]
    y_score: DataFrame, [samples x classes]
    """
    # TODO plot as grid per class
    for yc, pc in zip(y_true_df, y_score_df):
        # calc PAV per each class
        y = y_true_df[yc]
        p = y_score_df[pc]
        plot_rocch(y, p)

        # if plot_pav:
        #     n = len(y)
        #     segments = [[[i, y_sort[i]], [i, y_sort[i]]] for i in range(n)]
        #     lc = LineCollection(segments, zorder=0)
        #     lc.set_array(np.ones(len(y_sort)))
        #     lc.set_linewidths(0.5 * np.ones(n))
        #
        #     plt.figure()
        #     plt.plot(p_sort, y_sort, 'r.', markersize=12)
        #     plt.plot(p_sort, p_pav, 'g.-', markersize=12)
        #     plt.gca().add_collection(lc)
        #     plt.legend(('Data', 'Isotonic Fit'), loc='lower right')
        #     plt.title('Isotonic regression')
        #     plt.show()


def plot_rocch_fnr_fpr(y_true_df, y_score_df):
    pass


def mfom_smooth(y_true, y_score, alpha, beta):
    """
    return: smoothed FNR, FPR, class loss scores
    """
    pass


# ===
# discrete (or PAV calibrated) vs smoothed MFoM scores
# ===
# compare smooth and discrete scores, FN and FP


# check how alpha and betta of l_k affect the scores, plot

# ===
# ROC vs ROCCH
# ===

# convex hull ROC plot

# Isotonic regression or Platt calibration

# ===
# Whole set of scores vs batch: affect on the EER
# ===

# smEER, EER, pEER, AvgEER,

# whole scores vs batch subsampled scores

if __name__ == "__main__":

    # TODO: add command line interface

    P_df = TS.arr2DataFrame(TS.p_test)
    Y_df = TS.arr2DataFrame(TS.y_test)

    # toy_score_table(p_df=P_df, y_df=Y_df)
    class_wise_scatter(data_frame=P_df)

    # ===
    # score histograms
    # ===
    # pooled target/non-target
    tar_pool, ntar_pool = TS.pooled_tnt(p=P_df, y=Y_df)
    plot_histogram(tar_pool, ntar_pool, bins=10)

    # class-wise score split
    ts, nts = TS.class_wise_tnt(p=P_df, y=Y_df)
    class_wise_histograms(ts, nts)

    # ===
    # ROC curves
    # ===
    # pooled scores
    pool_sc = TS.pooled_scores(p=P_df, y=Y_df)
    y_true = pool_sc.values[:, 1]
    y_score = pool_sc.values[:, 0]
    plot_roc(y_true=y_true, y_score=y_score)

    # class-wise ROC curve
    class_wise_roc(Y_df, P_df)

    # ===
    # FNR vs FPR distributions
    # ===
    # pooled FNR vs FPR
    pool_sc = TS.pooled_scores(p=P_df, y=Y_df)
    y_true = pool_sc.values[:, 1]
    y_score = pool_sc.values[:, 0]
    plot_roc_fnr_fpr(y_true=y_true, y_score=y_score)

    # class-wise FNR vs FPR
    class_wise_roc_fnr_fpr(Y_df, P_df)

    # ===
    # ROCCH curves
    # ===
    # pooled
    # pool_sc = TS.pooled_scores(p=P_df, y=Y_df)
    # y_true = pool_sc.values[:, 1]
    # y_score = pool_sc.values[:, 0]
    # plot_roc(y_true=y_true, y_score=y_score)

    # class-wise
    # class_wise_roc(Y_df, P_df)

    class_wise_rocch(y_true_df=Y_df, y_score_df=P_df)

    # y_true = Y_df.values[:, 0]
    # y_score = P_df.values[:, 0]
    # n = len(Y_df.values[:, 0])
    #
    # ir = IsotonicRegression()
    # ids = np.argsort(y_score)
    # y_score_sr = np.sort(y_score)
    # y_true_sr = y_true[ids]
    #
    # y_pav = ir.fit_transform(y_score_sr, y_true_sr)
    #
    # segments = [[[i, y_true_sr[i]], [i, y_true_sr[i]]] for i in range(n)]
    # lc = LineCollection(segments, zorder=0)
    # lc.set_array(np.ones(len(y_true_sr)))
    # lc.set_linewidths(0.5 * np.ones(n))
    #
    # fig = plt.figure()
    # plt.plot(y_score_sr, y_true_sr, 'r.', markersize=12)
    # plt.plot(y_score_sr, y_pav, 'g.-', markersize=12)
    # plt.gca().add_collection(lc)
    # plt.legend(('Data', 'Isotonic Fit'), loc='lower right')
    # plt.title('Isotonic regression')
    # plt.show()


    # ===
    # ROCCH vs ROC FNR/FPR
    # ===


    # ===
    # PAV vs original scores distributions
    # ===


    # ===
    # smooth FN & FP (depend on alpha and beta) distributions and
    # value of smoothed EER and discrete EER on the right
    # ===
