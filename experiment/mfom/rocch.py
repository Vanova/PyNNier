"""
Compare discrete and convex hull ROC computation
and EER from these plots
"""

from matplotlib import pyplot as plt
import matplotlib as mpl
from pandas import *
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn import metrics


def colored_table(ax, vals, row_lab, col_lab):
    normal = mpl.colors.Normalize(vmin=vals.min() - .2, vmax=vals.max() + .2)
    ax.table(cellText=vals, rowLabels=row_lab, colLabels=col_lab,
             colWidths=[0.03] * vals.shape[1],
             cellColours=plt.cm.coolwarm(normal(vals)), alpha=0.5, loc='left')


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

idx = Index(range(1, 11))
p_test = DataFrame(p_test, index=idx, columns=['C_1', 'C_2', 'C_3', 'C_4'])
y_test = DataFrame(y_test, index=idx, columns=['C_1', 'C_2', 'C_3', 'C_4'])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2, 1, 1, xticks=[], yticks=[], frameon=False, )
colored_table(ax, vals=p_test.values, col_lab=p_test.columns, row_lab=p_test.index)

ax = fig.add_subplot(2, 1, 2, xticks=[], yticks=[], frameon=False)
colored_table(ax, vals=y_test.values, col_lab=y_test.columns, row_lab=y_test.index)
plt.show()

# Distributions
# df = pd.DataFrame(np.array(s_test), columns=['C_1', 'C_2', 'C_3', 'C_4'])
scatter_matrix(p_test, alpha=0.2, figsize=(6, 6), diagonal='kde')

# histograms plot
df4 = pd.DataFrame({'C_1_1': p_test['C_1'][y_test['C_1'] > 0].values, 'C_1_0': p_test['C_1'][y_test['C_1'] < 1].values}, columns=['C_1_0', 'C_1_1'])
df4.plot.hist(alpha=0.5, bins=10)
plt.show()

# 0. Isotonic regression

# 1. discrete ROC plot for tst scores
# fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, drop_intermediate=True)

# 2. PAV ROCCH algorithm

# 3. smEER and EER
