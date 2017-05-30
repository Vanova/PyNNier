import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import numpy as np
from scipy.stats import norm


def colored_table(ax, vals, row_lab, col_lab):
    """
    Plot colored table
    ax: axis of fig.add_subplot(...)
    vals: 2D array, numpy.ndarray
    row_lab: list of row indices
    col_lab: list of column indices
    """
    normal = mpl.colors.Normalize(vmin=vals.min() - .2, vmax=vals.max() + .2)
    ax.table(cellText=vals, rowLabels=row_lab, colLabels=col_lab,
             colWidths=[0.03] * vals.shape[1],
             cellColours=plt.cm.coolwarm(normal(vals)), alpha=0.5, loc='left')


def view_histogram(ax, tar, nontar, bins=5, gaus_fit=True, min=0., max=1.):
    """
    Return view of histogram on axes
    tar_pool: 1D array
    ntar_pool: 1D array
    gaus_fit: fit histogram with Gaussian
    ax: plot axes
    """
    ax.hist(tar, label='Target', bins=bins, alpha=0.5, color='b')
    ax.hist(nontar, label='Non-target', bins=bins, alpha=0.5, color='g')
    ax.legend(loc='upper right')

    if gaus_fit:
        # target scores 'best fit' line
        x = np.linspace(min, max, 100)
        (mu, s) = norm.fit(tar)
        y = norm.pdf(x, mu, s)
        ax.plot(x, y, alpha=0.3, color='b')

        # non-target scores 'best fit' line
        (mu, s) = norm.fit(nontar)
        y = norm.pdf(x, mu, s)
        ax.plot(x, y, '--', alpha=0.3, color='g')


def view_roc_curve(ax, fpr, tpr, roc_auc=None, color=None):
    lw = 2
    ax.plot(fpr, tpr, marker='o', linestyle='--', color=color, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(loc="lower right")

