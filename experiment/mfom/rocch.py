"""
Compare discrete and convex hull ROC computation
and EER from these plots
"""

from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from pandas.plotting import scatter_matrix
from scipy.stats import norm
import plotter
import toy_scores as TS
import numpy as np
import sklearn.metrics as metrics


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


def class_wise_roc(y_true_cw, y_score_cw):
    fprs, tprs, aucs = [], [], []

    n_classes = len(y_true_cw.columns)
    for c in range(n_classes):
        y_true = y_true_cw.values[:, c]
        y_score = y_score_cw.values[:, c]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, drop_intermediate=True)
        # roc_auc = metrics.auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(metrics.auc(fpr, tpr))

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


def pooled_roc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, drop_intermediate=True)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, marker='o', linestyle='--', color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def rocch():
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
    P_df = TS.arr2DataFrame(TS.p_test)
    Y_df = TS.arr2DataFrame(TS.y_test)

    # toy_score_table(p_df=P_df, y_df=Y_df)
    # class_wise_scatter(data_frame=P_df)

    # class-wise score split
    ts, nts = TS.class_wise_tnt(p=P_df, y=Y_df)
    # class_wise_histograms(ts, nts)

    # ===
    # score distributions
    # ===
    # pooled target/non-target
    tar_pool, ntar_pool = TS.pooled_tnt(p=P_df, y=Y_df)
    # pooled_histogram(tar_pool, ntar_pool, bins=10)


    # ===
    # class-wise discrete ROC plot
    # TODO: class-wise EER
    # ===
    class_wise_roc(Y_df, P_df)

    # ===
    # pooled discrete ROC plot
    # TODO: AvgEER
    # ===
    pool_sc = TS.pooled_scores(p=P_df, y=Y_df)
    y_true = pool_sc.values[:, 1]
    y_score = pool_sc.values[:, 0]
    pooled_roc(y_true=y_true, y_score=y_score)

    # ===
    # FNR vs FPR distributions
    # ===
    pool_sc = TS.pooled_scores(p=P_df, y=Y_df)
    y_true = pool_sc.values[:, 1]
    y_score = pool_sc.values[:, 0]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, drop_intermediate=True)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(thresholds, fpr, marker='o', linestyle='--', color='green', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(thresholds, 1-tpr, marker='o', linestyle='--', color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(thresholds, np.abs(1 - tpr - fpr), marker='o', linestyle='--', color='blue', label='ROC curve (area = %0.2f)' % roc_auc)

    # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


    # ===
    # pooled ROCCH
    # ===

    # ===
    # smooth FN & FP (depend on alpha and beta) distributions and
    # value of smoothed mF1 and discrete mF1 on the right
    # ===
