import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import colorConverter, ListedColormap


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
