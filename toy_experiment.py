import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from utils import toy_loader
from sklearn import preprocessing
from utils.plotters import NetworkVisualiser
from ann import network
from ann import mif_network

COLORS = np.array(['!',
                   '#FF3333',  # red [1, 0]
                   '#0198E1',  # blue [0, 1]
                   '#4DBD33',  # green [1, 1]
                   '#FCD116',  # yellow
                   '#FF7216',  # orange
                   '#BF5FFF',  # purple
                   '#87421F'  # brown
                   ])
MARKERS = np.array(['!',
                    '^',  # [1, 0]
                    'v',  # [0, 1]
                    'o'  # [1, 1]
                    ])
SET_TITLES = ['Training set',
              'Test set',
              'Dev set']


def plot_hyperplane(ax, clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    ax.plot(xx, yy, linestyle, label=label)


def plot_2d(figure_axes, x, y):
    # plot dots
    id_lbl = (y * [1, 2]).sum(axis=1)
    colors = COLORS.take(id_lbl)
    for c, m in zip(COLORS, MARKERS):
        xr = x[np.where(colors == c)]  # take class by color
        if len(xr):
            figure_axes.scatter(xr[:, 0], xr[:, 1], color=c, marker=m, edgecolor='black')

    # prepare hyperplanes
    # min_x = np.min(x[:, 0])
    # max_x = np.max(x[:, 0])
    #
    # classif = OneVsRestClassifier(SVC(kernel='linear'))
    # classif.fit(x, y)
    # plot_hyperplane(figure_axes, classif.estimators_[0], min_x, max_x, 'k--',
    #                 'Boundary\nfor class 1')
    # plot_hyperplane(figure_axes, classif.estimators_[1], min_x, max_x, 'k-.',
    #                 'Boundary\nfor class 2')


def plot_3d(figure_axes, x, y):
    id_lbl = (y * [1, 2]).sum(axis=1)
    colors = COLORS.take(id_lbl)
    for c, m in zip(COLORS, MARKERS):
        xr = x[np.where(colors == c)]
        if len(xr):
            figure_axes.scatter(xr[:, 0], xr[:, 1], xr[:, 2], color=c, marker=m, edgecolor='b')


def plot_toy_data(data_list, titles, feature_dim):
    # TODO feat dim 2D or 3D
    # num_set = len(data_list)
    # dim = data_list[0].shape(0)
    if feature_dim == 2:
        fig, _ = plt.subplots(nrows=1, ncols=len(data_list),
                              sharex='row', sharey='row', figsize=(15, 6))
        for i, ax in enumerate(fig.axes):
            ax.set_title(titles[i])
            tr_x, tr_y = toy_loader.plot_format(data_list[i])
            plot_2d(ax, tr_x, tr_y)
        plt.show()
    if feature_dim == 3:
        fig = plt.figure(figsize=(12, 5))
        for i in xrange(len(data_list)):
            ax = fig.add_subplot(1, len(data_list), i + 1, projection='3d')
            ax.set_title(titles[i])
            tr_x, tr_y = toy_loader.plot_format(data_list[i])
            plot_3d(ax, tr_x, tr_y)
        plt.show()
    if feature_dim > 3:
        print("Not implemented...")


def data_stats(data):
    feats, t = toy_loader.plot_format(data)
    print("Mean stats: {0}".format(np.mean(feats, axis=0)))
    print("Std. stats: {0}".format(np.std(feats, axis=0)))
    print("Min: {0}".format(np.min(feats, axis=0)))
    print("Max: {0}".format(np.max(feats, axis=0)))
    plt.figure(figsize=(3, 4))
    plt.boxplot(feats)
    plt.show()


# TODO:
# + implement NN saving
# 2. plot model weights
# 2.1. plot data before and after classification
# 2.2. Plot value of d_k based on the outputs of sigma
# + plot toy 2D data: plot_toy
# + plot toy 3D data: plot_toy
# + plot loss surface
# + simple architecture [2, 2] or [3, 3]
# 5. make video of parameters optimisation

# Plot data scatter distribution
feature_dim = 2
# train_data, dev_data, test_data = toy_loader.load_data(n_tr=250, n_dev=50, n_tst=50,
#                                                        n_features=feature_dim, n_classes=2)
train_data, dev_data, test_data = toy_loader.load_data(n_tr=250, n_dev=50, n_tst=50,
                                                       n_features=feature_dim, n_classes=2,
                                                       scaler=preprocessing.StandardScaler())
plot_toy_data([train_data, dev_data, test_data], SET_TITLES, feature_dim)
data_stats(train_data)
#####
# Simple toy experiment
# 1.MSE loss and Sigmoid outputs
#####
epochs = 100
mini_batch = 10
learn_rate = 0.1
architecture = [feature_dim, 2]
network = network.Network(architecture)
eval, loss, list_ws = network.SGD(train_data, epochs, mini_batch, learn_rate,
                                  test_data=test_data, is_list_weights=True)
best_weights = network.weights
print("mF1 on test set: {0}".format(eval[-1]))
print("MSE loss: {0}".format(loss[-1]))
print("Network optimal weights:")
print(best_weights)
#####
# visualize the network weights
#####
net_viz = NetworkVisualiser()
net_viz.plot_neurons_cost_surface(network, train_data, title="The MSE error surface")
# visualize decision boundary
network.weights = best_weights
net_viz.plot_decision_boundaries(network, train_data, title="The MSE error surface")


# file_net = "./data/experiment/toy/toy_epo_{0}_btch_{1}_lr_{2}". \
#     format(epochs, mini_batch, learn_rate)
# net2.save(file_net)
# net2 = network.load(file_net)

## 2.MFoM network with Sigmoid outputs
# epochs = 100
# net = mif_network.MifNetwork(architecture, alpha=5., beta=0)
# eval, loss = net.SGD(train_data, epochs, mini_batch, learn_rate, test_data=test_data)


# # plot nnet curve surface
# fig = plt.figure()
# # ax = fig.gca(projection='3d')
# gridN = 10
# nx, ny = (gridN, gridN)
# x = np.linspace(-5, 70, nx)
# y = np.linspace(-5, 70, ny)
# X, Y = np.meshgrid(x, y)
#
# x = X.reshape((gridN*gridN,1))
# y = Y.reshape((gridN*gridN,1))
# gdata = [np.array([a.tolist(), b.tolist()]) for a,b in zip(x, y)]
#
# # out = [(np.argmax(net2.feedforward(x)), np.argmax(y))
# #        for (x, y) in test_data]
#
# out = [net2.feedforward(a) for a in gdata]
# out = toy_loader.plot_format_no_ticks(out)
#
# out1 = [a[0] for a in out]
# out1 = np.array(out1)
# out1 = out1.reshape((gridN, gridN))
#
#
# CS = plt.contourf(X, Y, out1, 10,
#                   #[-1, -0.1, 0, 0.1],
#                   #alpha=0.5,
#                   cmap=plt.cm.bone)
# # Make a colorbar for the ContourSet returned by the contourf call.
# cbar = plt.colorbar(CS)
# plt.show()
#
#
