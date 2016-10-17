import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from utils import toy_loader
from sklearn import preprocessing
import utils.plotters as viz
from ann import network
from ann import mif_network

SET_TITLES = ['Training set',
              'Test set',
              'Dev set']

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
            viz.plot_2d_scatter(ax, tr_x, tr_y)
        plt.show()
    if feature_dim == 3:
        fig = plt.figure(figsize=(12, 5))
        for i in xrange(len(data_list)):
            ax = fig.add_subplot(1, len(data_list), i + 1, projection='3d')
            ax.set_title(titles[i])
            tr_x, tr_y = toy_loader.plot_format(data_list[i])
            viz.plot_3d_scatter(ax, tr_x, tr_y)
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
# + plot data before and after classification
# 2.2. Plot value of d_k based on the outputs of sigma
# + plot toy 2D data: plot_toy
# + plot toy 3D data: plot_toy
# + plot loss surface
# + simple architecture [2, 2] or [3, 3]

#####
# Prepare data and plot data scatter distribution
#####
feature_dim = 2
train_data, dev_data, test_data = toy_loader.load_data(n_tr=250, n_dev=50, n_tst=50,
                                                       n_features=feature_dim, n_classes=2,
                                                       scaler=preprocessing.StandardScaler())
# plot_toy_data([train_data, dev_data, test_data], SET_TITLES, feature_dim)
# data_stats(train_data)

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
best_weights = copy.deepcopy(network.weights)
print("mF1 on test set: {0}".format(eval[-1]))
print("MSE loss: {0}".format(loss[-1]))
print("Network optimal weights:")
print(best_weights[-1])
#####
# visualize the network weights
#####
net_viz = viz.NetworkVisualiser()
net_viz.plot_neurons_cost_surface(network, train_data, title="The MSE error surface")
# visualize decision boundary
network.weights = best_weights
net_viz.plot_decision_boundaries(network, train_data, title="Decision surface")


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
