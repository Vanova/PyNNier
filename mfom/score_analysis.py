"""
Compare discrete and convex hull ROC computation
and EER from these plots
"""
import numpy as np
import sklearn.metrics as sk_metrics
from pandas.plotting import scatter_matrix
from scipy.stats import norm
import matplotlib.pyplot as plt
from mfom.utils import toy_scores as TS, plotter
from metrics import eer, sklearn_rocch


def toy_score_table(p_df, y_df):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 1, 1, xticks=[], yticks=[], frameon=False)
    plotter.colored_table(ax, vals=p_df.values, col_lab=p_df.columns, row_lab=p_df.index)

    ax = fig.add_subplot(2, 1, 2, xticks=[], yticks=[], frameon=False)
    plotter.colored_table(ax, vals=y_df.values, col_lab=y_df.columns, row_lab=y_df.index)
    plt.show()


def class_wise_scatter(data_frame):
    scatter_matrix(data_frame, alpha=0.5, figsize=(6, 6), diagonal='kde')
    plt.show()


def class_wise_histograms(ts, nts, bins=10):
    # Histograms plots
    fig = plt.figure(figsize=plt.figaspect(0.5))
    for c in range(len(ts)):
        ax = fig.add_subplot(2, 2, c + 1, frameon=True)
        ax.hist(ts[c].values, label=ts[c].columns[0], bins=bins, alpha=0.5, color='b')
        ax.hist(nts[c].values, label=nts[c].columns[0], bins=bins, alpha=0.5, color='g')

        # target scores 'best fit' line
        (mu, s) = norm.fit(ts[c].values)
        x = np.linspace(0., 1., 100)
        y = norm.pdf(x, mu, s)
        ax.plot(x, y, alpha=0.3, color='b')
        # non-target scores 'best fit' line
        (mu, s) = norm.fit(nts[c].values)
        y = norm.pdf(x, mu, s)
        ax.plot(x, y, '--', alpha=0.3, color='g')
        ax.legend(loc='upper right')

    fig.tight_layout()
    plt.show()


def pooled_histogram(tar_pool, ntar_pool, bins=5):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    ax.hist(tar_pool, label='Target', bins=bins, alpha=0.5, color='b')
    ax.hist(ntar_pool, label='Non-target', bins=bins, alpha=0.5, color='g')

    # target scores 'best fit' line
    x = np.linspace(0., 1., 100)
    (mu, s) = norm.fit(tar_pool)
    y = norm.pdf(x, mu, s)
    ax.plot(x, y, alpha=0.3, color='b')
    # non-target scores 'best fit' line
    (mu, s) = norm.fit(ntar_pool)
    y = norm.pdf(x, mu, s)
    ax.plot(x, y, '--', alpha=0.3, color='g')
    ax.legend(loc='upper right')

    fig.tight_layout()
    plt.show()


def plot_roc(y_true, y_score):
    """
    y_true: 1D array
    y_score: 1D array
    """
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=True)
    roc_auc = sk_metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, marker='o', linestyle='--', color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_class_wise_roc(y_true_cw, y_score_cw):
    """
    y_true_cw: DataFrame, [samples x classes]
    y_score_cw: DataFrame, [samples x classes]
    """
    fprs, tprs, aucs = [], [], []
    n_classes = len(y_true_cw.columns)
    for c in range(n_classes):
        y_true = y_true_cw.values[:, c]
        y_score = y_score_cw.values[:, c]
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=True)
        # roc_auc = metrics.auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(sk_metrics.auc(fpr, tpr))

    for c in range(n_classes):
        plt.plot(fprs[c], tprs[c], marker='o', linestyle='--',
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(c + 1, aucs[c]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_fnr_fpr(y_true, y_score):
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=True)
    fpr = np.insert(fpr, 0, 0.)
    tpr = np.insert(tpr, 0, 0.)
    fnr = 1. - tpr
    thresholds = np.insert(thresholds, 0, 1.)
    eer_val = eer(y_true, y_score)

    plt.figure()
    plt.plot(thresholds, fpr, marker='o', linestyle='--', label='FPR')
    plt.plot(thresholds, fnr, marker='o', linestyle='--', label='FNR')
    plt.plot(thresholds, np.abs(fnr - fpr), marker='.', linestyle=':', alpha=0.8, lw=1, label='|FNR - FPR|')
    plt.plot(thresholds, np.abs(fnr + fpr), marker='.', linestyle=':', alpha=0.8, lw=1, label='|FNR + FPR|')
    id_x = np.argmin(np.abs(fnr - fpr))
    plt.plot(thresholds[id_x], eer_val, linestyle=' ', marker='*', markersize=10, label='pEER', color='red')
    plt.xlabel('Thresholds')
    plt.ylabel('Error rate')
    plt.legend(loc="center right")
    plt.show()


def plot_rocch(y_true, y_score):
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


def plot_class_wise_rocch(y_true_df, y_score_df):
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
    # class_wise_scatter(data_frame=P_df)

    # class-wise score split
    # ts, nts = TS.class_wise_tnt(p=P_df, y=Y_df)
    # class_wise_histograms(ts, nts)

    # ===
    # score distributions
    # ===
    # pooled target/non-target
    tar_pool, ntar_pool = TS.pooled_tnt(p=P_df, y=Y_df)
    # pooled_histogram(tar_pool, ntar_pool, bins=10)


    # ===
    # class-wise discrete ROC plot
    # TODO: class-wise EER, AvgEER
    # ===
    # class_wise_roc(Y_df, P_df)

    # ===
    # pooled discrete ROC plot
    # TODO: pooled EER
    # ===
    pool_sc = TS.pooled_scores(p=P_df, y=Y_df)
    y_true = pool_sc.values[:, 1]
    y_score = pool_sc.values[:, 0]
    # pooled_roc(y_true=y_true, y_score=y_score)

    # ===
    # class-wise FNR vs FPR distributions
    # ===
    # TODO: implement
    # pool_sc = TS.pooled_scores(p=P_df, y=Y_df)
    # y_true = pool_sc.values[:, 1]
    # y_score = pool_sc.values[:, 0]
    # pooled_fnr_fpr(y_true=y_true, y_score=y_score)

    # ===
    # pooled FNR vs FPR distributions
    # ===
    pool_sc = TS.pooled_scores(p=P_df, y=Y_df)
    y_true = pool_sc.values[:, 1]
    y_score = pool_sc.values[:, 0]
    # pooled_fnr_fpr(y_true=y_true, y_score=y_score)

    # ===
    # class-wise Isotonic regression (sklearn), i.e. ROCCH
    # ===
    plot_class_wise_rocch(y_true_df=Y_df, y_score_df=P_df)

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
    # smooth FN & FP (depend on alpha and beta) distributions and
    # value of smoothed mF1 and discrete mF1 on the right
    # ===
