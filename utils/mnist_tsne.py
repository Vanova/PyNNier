import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import numpy as np
import gzip, cPickle
from tsne import bh_sne
import plotters as pltr

# n_groups = 4
# bar_width = 0.35
# opacity = 0.8
# index = np.arange(n_groups)
# fig = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig.add_subplot(1, 1, 1)
#
# a = [90, 55, 40, 65]
# b = [85, 62, 54, 20]
# g1 = ax.bar(index, a, bar_width, alpha=opacity, color='b', label='Fold1')
# g2 = ax.bar(index + bar_width, b, bar_width, alpha=opacity, color='g', label='Fold2')
#
# ax.set_xlabel('Tags')
# ax.set_ylabel('# of tags')
# ax.set_xticks(index + bar_width)
# ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
# # ax.set_xticks(index + bar_width, ('A', 'B', 'C', 'D'))
# # ax.set_grid(True)
#
# plt.show()

f = gzip.open("mnist.pkl.gz", "rb")
train, val, test = cPickle.load(f)
f.close()

X = np.asarray(np.vstack((train[0], val[0], test[0])), dtype=np.float64)
y = np.hstack((train[1], val[1], test[1]))

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# image_size = np.shape(X_train[0])[1]
# X_train /= 255
# X_test /= 255
# X = np.asarray(np.vstack((X_train, X_test)), dtype=np.float64)
# y = np.hstack((y_train, y_test))
X_2d = bh_sne(X)

pltr.show_scatter(X_2d[:, 0], X_2d[:, 1])