"""
Compare discrete and convex hull ROC computation
and EER from these plots
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sk_metrics
from matplotlib.pyplot import cm
from pandas.plotting import scatter_matrix
import cost_function as mfom_cost
import mfom.utils.dcase_scores as mfom_dcase
import mfom.visual.plotter as mfom_plt
import mfom.utils.toy_scores as toy_sc
from metrics.metrics import eer, sklearn_rocch, sklearn_pav


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
    # calculate
    fpr, tpr, _, y_calibr, p_calibr = sklearn_rocch(y_true, y_score)
    roc_auc = sk_metrics.auc(fpr, tpr)
    eer_val = eer(y_true=y_calibr, y_score=p_calibr)
    # plot ROCCH
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    mfom_plt.view_roc_curve(ax, fpr, tpr, roc_auc=roc_auc, eer_val=eer_val, title='ROC convex hull')
    fig.tight_layout()
    plt.show()


def plot_rocch_fnr_fpr(y_true, y_score):
    """
    y_true: 1D array
    y_score: 1D array
    """
    # calculate ROCCH and calibrated scores
    fpr, tpr, thresholds, y_calibr, p_calibr = sklearn_rocch(y_true, y_score)
    eer_val = eer(y_true=y_calibr, y_score=p_calibr)
    fpr = np.insert(fpr, 0, 0.)
    tpr = np.insert(tpr, 0, 0.)
    fnr = 1. - tpr
    thresholds = np.insert(thresholds, 0, 1.)

    # plot FNR/FPR distributions
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    mfom_plt.view_fnr_fpr_dist(ax, fnr, fpr, thresholds, eer_val)
    fig.tight_layout()
    fig.savefig('./misc/fnr_fpr_distr.png', dpi=200)
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


def class_wise_roc(y_true_df, y_score_df):
    """
    y_true_df: DataFrame, [samples x classes]
    y_score_df: DataFrame, [samples x classes]
    """
    # calculate fpr/tpr per class
    fprs, tprs, aucs, eer_vals = [], [], [], []
    for yc, pc in zip(y_true_df, y_score_df):
        y_true = y_true_df[yc].values
        y_score = y_score_df[pc].values
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=True)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(sk_metrics.auc(fpr, tpr))
        eer_vals.append(eer(y_true, y_score))

    # plot curves
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    n_classes = len(y_true_df.columns)
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
    y_true_df: DataFrame, [samples x classes]
    y_score_df: DataFrame, [samples x classes]
    """
    # calculate fpr/tpr per class
    fprs, tprs, aucs, eer_vals = [], [], [], []
    for yc, pc in zip(y_true_df, y_score_df):
        y_true = y_true_df[yc].values
        y_score = y_score_df[pc].values
        fpr, tpr, _, y_calibr, p_calibr = sklearn_rocch(y_true, y_score)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(sk_metrics.auc(fpr, tpr))
        eer_vals.append(eer(y_calibr, p_calibr))

    # plot ROCCH curves
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    n_classes = len(y_true_df.columns)
    color = iter(cm.rainbow(np.linspace(0, 1, n_classes)))
    for c in range(n_classes):
        mfom_plt.view_roc_curve(ax, fprs[c], tprs[c], roc_auc=aucs[c], eer_val=eer_vals[c],
                                color=next(color), title='ROC convex hull')
    fig.tight_layout()
    plt.show()


def class_wise_rocch_fnr_fpr(y_true_df, y_score_df):
    """
    y_true_df: DataFrame, [samples x classes]
    y_score_df: DataFrame, [samples x classes]
    """
    # calculate fpr/tpr per class
    fprs, fnrs, thresholds, eer_vals = [], [], [], []
    for cname in y_true_df:
        y_true = y_true_df[cname].values
        y_score = y_score_df[cname].values
        fpr, tpr, thresh, y_calibr, p_calibr = sklearn_rocch(y_true, y_score)
        # TODO: fix plot values
        fnr = 1. - tpr
        # append for each class
        fprs.append(fpr)
        fnrs.append(fnr)
        thresholds.append(thresh)
        eer_vals.append(eer(y_true=y_calibr, y_score=p_calibr))

    # plot FNR/FPR distributions
    fig = plt.figure(figsize=plt.figaspect(0.5))
    n_row = 2
    n_col = np.round(len(y_true_df.columns) / float(n_row))
    i = 1
    for fnr, fpr, th, er in zip(fnrs, fprs, thresholds, eer_vals):
        ax = fig.add_subplot(n_row, n_col, i, frameon=True)
        mfom_plt.view_fnr_fpr_dist(ax, fnr, fpr, th, er)
        i += 1
    fig.tight_layout()
    plt.show()


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


if __name__ == "__main__":
    debug = False
    # TODO REFACTORRRRR!!!!
    # P_df = toy_sc.arr2DataFrame(toy_sc.p_test)
    # Y_df = toy_sc.arr2DataFrame(toy_sc.y_test)
    P_df = mfom_dcase.read_dcase('data/test_scores/results_fold2.txt')
    Y_df = mfom_dcase.read_dcase('data/test_scores/y_true_fold2.txt')

    # 1 - l_k scores
    loss_scores = mfom_cost._uvz_loss_scores(y_true=Y_df.values, y_pred=P_df.values, alpha=3.)
    ls_df = toy_sc.arr2DataFrame(1. - loss_scores, row_id=P_df.index, col_id=P_df.columns)

    # TODO ROC per each fold on one plot
    fig = plt.figure()
    for i in range(1, 6):
        P_df = mfom_dcase.read_dcase('data/test_scores/results_fold%d.txt' % i)
        Y_df = mfom_dcase.read_dcase('data/test_scores/y_true_fold%d.txt' % i)
        loss_scores = mfom_cost._uvz_loss_scores(y_true=Y_df.values, y_pred=P_df.values, alpha=3.)
        ls_df = toy_sc.arr2DataFrame(1. - loss_scores, row_id=P_df.index, col_id=P_df.columns)
        # loss_scores = mfom_cost._uvz_loss_scores(y_true=Y_df.values, y_pred=ls_df.values, alpha=3.)
        # ls_df = toy_sc.arr2DataFrame(1. - loss_scores, row_id=P_df.index, col_id=P_df.columns)
        # ===
        # ROC curves
        # ===
        # pooled scores
        # y_score, y_true = toy_sc.pool_scores(p_df=P_df, y_df=Y_df)
        y_score, y_true = toy_sc.pool_scores(p_df=ls_df, y_df=Y_df)
        # calculate ROC
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=True)
        roc_auc = sk_metrics.auc(fpr, tpr)
        eer_val = eer(y_true=y_true, y_score=y_score)

        ax = fig.add_subplot(1, 1, 1, frameon=True)
        mfom_plt.view_roc_curve(ax, fpr, tpr, roc_auc=roc_auc, eer_val=eer_val, label='Fold#%d (area = %0.2f)' % (i, roc_auc))
    fig.tight_layout()
    fig.savefig('./misc/roc_mfom.png', dpi=200)
    plt.show()


    if not debug:
        # toy_score_table(p_df=P_df, y_df=Y_df)
        class_wise_scatter(data_frame=P_df)

        # ===
        # score histograms
        # ===
        # pooled target/non-target
        tar, ntar = toy_sc.pool_split_tnt(p_df=P_df, y_df=Y_df)
        plot_histogram(tar, ntar, bins=10)
        # class-wise score split
        ts, nts = toy_sc.class_wise_tnt(p=P_df, y=Y_df)
        class_wise_histograms(ts, nts)

        # ===
        # ROC curves
        # ===
        # pooled scores
        y_score, y_true = toy_sc.pool_scores(p_df=P_df, y_df=Y_df)
        plot_roc(y_true=y_true, y_score=y_score)
        # class-wise ROC curve
        class_wise_roc(Y_df, P_df)

        # ===
        # FNR vs FPR distributions
        # ===
        # pooled scores
        y_score, y_true = toy_sc.pool_scores(p_df=P_df, y_df=Y_df)
        plot_roc_fnr_fpr(y_true=y_true, y_score=y_score)
        # class-wise scores
        class_wise_roc_fnr_fpr(Y_df, P_df)

        # ===
        # ROCCH curves
        # ===
        # pooled scores
        y_score, y_true = toy_sc.pool_scores(p_df=P_df, y_df=Y_df)
        plot_rocch(y_true=y_true, y_score=y_score)
        # class-wise scores
        class_wise_rocch(y_true_df=Y_df, y_score_df=P_df)

        # ===
        # ROCCH FNR/FPR distributions
        # ===
        y_score, y_true = toy_sc.pool_scores(p_df=P_df, y_df=Y_df)
        plot_rocch_fnr_fpr(y_true=y_true, y_score=y_score)
        # class-wise scores
        class_wise_rocch_fnr_fpr(y_true_df=Y_df, y_score_df=P_df)

        # ===
        # PAV vs original scores distributions: hist
        # ===
        # how PAV changes the POOLED score distributions,
        # even if these are correlated classes
        # original pool scores
        tar, ntar = toy_sc.pool_split_tnt(p_df=P_df, y_df=Y_df)
        plot_histogram(tar, ntar, bins=10)

        # calibrated pool scores
        y_score_df, y_true_df = toy_sc.pool_scores(p_df=P_df, y_df=Y_df)
        y_calibr, p_calibr = sklearn_pav(y_true=y_true_df.values, y_score=y_score_df.values)
        # split up on target/non-target scores
        tar, ntar = toy_sc.array_tnt(p=p_calibr, y=y_calibr)
        # plot calibrated
        plot_histogram(tar, ntar, bins=10)

        # class-wise calibration
        y_cal_df, p_cal_df = toy_sc.calibrate_scores(p_df=P_df, y_df=Y_df)
        ts, nts = toy_sc.class_wise_tnt(p=p_cal_df, y=y_cal_df)
        class_wise_histograms(ts, nts)

        # # ===
        # # Histograms: original scores
        # # ===
        # # pooled scores
        # tar, ntar = TS.pool_split_tnt(p_df=P_df, y_df=Y_df)
        # plot_histogram(tar, ntar, bins=10)
        # # class-wise score split
        # ts, nts = TS.class_wise_tnt(p=P_df, y=Y_df)
        # class_wise_histograms(ts, nts)
        #
        # # ===
        # # Histograms: MFoM scores
        # # ===
        # # pooled scores
        # tar, ntar = TS.pool_split_tnt(p_df=ls_df, y_df=Y_df)
        # plot_histogram(tar, ntar, bins=10)
        # # class-wise score split
        # ts, nts = TS.class_wise_tnt(p=ls_df, y=Y_df)
        # class_wise_histograms(ts, nts)
        #
        # # ===
        # # ROC: original
        # # ===
        # # original
        # y_score, y_true = TS.pool_scores(p_df=P_df, y_df=Y_df)
        # plot_roc(y_true=y_true.values, y_score=y_score)
        # # class-wise ROC curve
        # class_wise_roc(Y_df, P_df)
        #
        # # ===
        # # ROCCH: original scores
        # # ===
        # # pooled scores
        # y_score, y_true = TS.pool_scores(p_df=P_df, y_df=Y_df)
        # plot_rocch(y_true=y_true, y_score=y_score)
        # # class-wise scores
        # class_wise_rocch(y_true_df=Y_df, y_score_df=P_df)

        # ===
        # ROC: MFoM
        # ===
        # pooled scores
        y_score, y_true = toy_sc.pool_scores(p_df=ls_df, y_df=Y_df)
        plot_roc(y_true=y_true, y_score=y_score)
        # class-wise ROC curve
        # class_wise_roc(Y_df, ls_df)

        # 2nd MFoM
        loss_scores = mfom_cost._uvz_loss_scores(y_true=Y_df.values, y_pred=ls_df.values, alpha=3.)
        ls_df = toy_sc.arr2DataFrame(1. - loss_scores, row_id=P_df.index, col_id=P_df.columns)

        y_score, y_true = toy_sc.pool_scores(p_df=ls_df, y_df=Y_df)
        plot_roc(y_true=y_true, y_score=y_score)

        # ===
        # Histograms: MFoM scores
        # ===
        # pooled scores
        tar, ntar = toy_sc.pool_split_tnt(p_df=ls_df, y_df=Y_df)
        plot_histogram(tar, ntar, bins=10)
        # class-wise score split
        ts, nts = toy_sc.class_wise_tnt(p=ls_df, y=Y_df)
        class_wise_histograms(ts, nts)


        # 3d MfoM
        loss_scores = mfom_cost._uvz_loss_scores(y_true=Y_df.values, y_pred=ls_df.values, alpha=3.)
        ls_df = toy_sc.arr2DataFrame(1. - loss_scores, row_id=P_df.index, col_id=P_df.columns)

        y_score, y_true = toy_sc.pool_scores(p_df=ls_df, y_df=Y_df)
        plot_roc(y_true=y_true, y_score=y_score)

        # ===
        # Histograms: MFoM scores
        # ===
        # pooled scores
        tar, ntar = toy_sc.pool_split_tnt(p_df=ls_df, y_df=Y_df)
        plot_histogram(tar, ntar, bins=10)
        # class-wise score split
        ts, nts = toy_sc.class_wise_tnt(p=ls_df, y=Y_df)
        class_wise_histograms(ts, nts)

        # 4th MfoM
        loss_scores = mfom_cost._uvz_loss_scores(y_true=Y_df.values, y_pred=ls_df.values, alpha=3.)
        ls_df = toy_sc.arr2DataFrame(1. - loss_scores, row_id=P_df.index, col_id=P_df.columns)

        y_score, y_true = toy_sc.pool_scores(p_df=ls_df, y_df=Y_df)
        plot_roc(y_true=y_true, y_score=y_score)

        # ===
        # ROCCH: MFoM scores
        # ===
        # pooled scores
        y_score, y_true = toy_sc.pool_scores(p_df=ls_df, y_df=Y_df)
        plot_rocch(y_true=y_true, y_score=y_score)
        # class-wise scores
        class_wise_rocch(y_true_df=Y_df, y_score_df=ls_df)

        # ===
        # Histograms: MFoM scores
        # ===
        # pooled scores
        tar, ntar = toy_sc.pool_split_tnt(p_df=ls_df, y_df=Y_df)
        plot_histogram(tar, ntar, bins=10)
        # class-wise score split
        ts, nts = toy_sc.class_wise_tnt(p=ls_df, y=Y_df)
        class_wise_histograms(ts, nts)

        # ===
        # Histograms: ground truth
        # ===
        # pooled scores
        tar, ntar = toy_sc.pool_split_tnt(p_df=Y_df, y_df=Y_df)
        plot_histogram(tar, ntar, bins=10)
        # class-wise score split
        ts, nts = toy_sc.class_wise_tnt(p=Y_df, y=Y_df)
        class_wise_histograms(ts, nts)


        # ===
        # smooth FN & FP (depends on alpha and beta) distributions and
        # value of smoothed EER and discrete EER
        # ===


        # ===
        # Whole set of scores vs batches: affect on the EER, smEER, pEER, AvgEER
        # ===


        # ===
        # Isotonic regression or Platt calibration
        # ===