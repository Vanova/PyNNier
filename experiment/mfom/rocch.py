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
import pandas as pd

# load scores
P_df = TS.arr2DataFrame(TS.p_test)
Y_df = TS.arr2DataFrame(TS.y_test)

# Score table
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2, 1, 1, xticks=[], yticks=[], frameon=False)
plotter.colored_table(ax, vals=P_df.values, col_lab=Y_df.columns, row_lab=P_df.index)

ax = fig.add_subplot(2, 1, 2, xticks=[], yticks=[], frameon=False)
plotter.colored_table(ax, vals=Y_df.values, col_lab=Y_df.columns, row_lab=Y_df.index)
plt.show()

# Scatter distributions
scatter_matrix(P_df, alpha=0.5, figsize=(6, 6), diagonal='kde')
plt.show()

# Histograms plots
ts, nts = TS.class_wise_tnt(p=P_df, y=Y_df)

fig = plt.figure(figsize=plt.figaspect(0.5))
bins = 10
for c in range(len(ts)):
    ax = fig.add_subplot(2, 2, c+1, frameon=True)
    n1, bins1, pat1 = ax.hist(ts[c].values, label=ts[c].columns[0], bins=bins, alpha=0.5, color='b')
    n2, bins2, pat2 = ax.hist(nts[c].values, label=nts[c].columns[0], bins=bins, alpha=0.5, color='g')

    # add a 'best fit' line
    x = np.linspace(0., 1., 100)
    (mu1, s1) = norm.fit(ts[c].values)
    y = norm.pdf(x, mu1, s1)
    # y = mlab.normpdf(bins1, mu1, s1)
    ax.plot(x, y, alpha=0.3, color='b')

    (mu2, s2) = norm.fit(nts[c].values)
    y = norm.pdf(x, mu2, s2)
    ax.plot(x, y, '--', alpha=0.3, color='g')
    ax.legend(loc='upper right')

fig.tight_layout()
plt.show()


# ===
# score distributions
# ===
# scores distributions: pooled, target/non-target, FNR vs FPR

# TS.pooled_scores(p=P_df, y=Y_df)
tar_pool, ntar_pool = TS.pooled_tnt(p=P_df, y=Y_df)
fig = plt.figure(figsize=plt.figaspect(0.5))
bins = 5

ax = fig.add_subplot(1, 1, 1, frameon=True)
n1, bins1, pat1 = ax.hist(tar_pool, label='Target', bins=bins, alpha=0.5, color='b')
n2, bins2, pat2 = ax.hist(ntar_pool, label='Non-target', bins=bins, alpha=0.5, color='g')

# add a 'best fit' line
# xmin, xmax = ax.get_xlim()
x = np.linspace(0., 1., 100)
(mu1, s1) = norm.fit(tar_pool)
y = norm.pdf(x, mu1, s1)
ax.plot(x, y, alpha=0.3, color='b')

(mu2, s2) = norm.fit(ntar_pool)
y = norm.pdf(x, mu2, s2)
ax.plot(x, y, '--', alpha=0.3, color='g')
ax.legend(loc='upper right')

fig.tight_layout()
plt.show()

# ===
# discrete (or PAV calibrated) vs smoothed MFoM scores
# ===
# compare smooth and discrete scores, FN and FP


# check how alpha and betta of l_k affect the scores, plot

# ===
# ROC vs ROCCH
# ===

# discrete ROC plot
# fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, drop_intermediate=True)

# convex hull ROC plot

# Isotonic regression or Platt calibration

# ===
# Whole set of scores vs batch: affect on the EER
# ===

# smEER, EER, pEER, AvgEER,

# whole scores vs batch subsampled scores
