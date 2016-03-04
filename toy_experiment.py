import numpy as np
import matplotlib.pyplot as plt
from plot_multilabel import plot_subfigure
from sklearn.datasets import make_multilabel_classification


def plot_2d():


#def plot_3d():

# TODO:
# 1. implement NN saving
# 2. plot model weights
# 3. simple architecture [2, 2] or [3, 3]

plt.figure(figsize=(8, 6))

X, Y, p_c, p_w_c = make_ml_clf(n_samples=150, n_features=2,
                                   n_classes=n_classes, n_labels=n_labels,
                                   length=length, allow_unlabeled=False,
                                   return_distributions=True,
                                   random_state=RANDOM_SEED)



X, Y = make_multilabel_classification(n_features=2, n_classes=2, n_labels=1,
                                      length= 300,
                                      allow_unlabeled=False,
                                      random_state=1)
plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")

