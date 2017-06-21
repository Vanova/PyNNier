"""
Sigmoid and MFoM scores, t-SNE plots
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mfom.utils import toy_scores as toy_sc
import mfom.utils.dcase_scores as mfom_dcase
import cost_function as mfom_cost
import matplotlib.cm as cm

# Scale and visualize the embedding vectors
def plot_embedding(X, labels, title=None):
    """
    X: 2D array, [samples x dim]
    labels: list of strings, [samples]
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    # lab_list = sorted(list(set(labels))) # strings
    lab_list = list(set(labels))

    plt.figure()
    ax = plt.subplot(111)

    colors = cm.gist_ncar(np.linspace(0, 1, len(lab_list)))
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], color=colors[lab_list.index(labels[i])])

    plt.xticks([]), plt.yticks([])
    plt.show()
    if title is not None:
        plt.title(title)


P_df = mfom_dcase.read_dcase('data/test_scores/results_fold2.txt')
Y_df = mfom_dcase.read_dcase('data/test_scores/y_true_fold2.txt')

# 1 - l_k scores
loss_scores = mfom_cost._uvz_loss_scores(y_true=Y_df.values, y_pred=P_df.values, alpha=3.)
ls_df = toy_sc.arr2DataFrame(1. - loss_scores, row_id=P_df.index, col_id=P_df.columns)

# loss_scores = mfom_cost._uvz_loss_scores(y_true=Y_df.values, y_pred=ls_df.values, alpha=3.)
# ls_df = toy_sc.arr2DataFrame(1. - loss_scores, row_id=P_df.index, col_id=P_df.columns)

model = TSNE(n_components=2, init='pca', learning_rate=300, perplexity=30, verbose=2, angle=0.1, random_state=777)
trans = model.fit_transform(P_df.values)
str_labs = [''.join(Y_df.columns[Y_df.values[i] == True].format()) for i in range(len(Y_df))]
plot_embedding(trans, str_labs)
