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


# - Plot value of d_k based on the outputs of sigma
# + implement NN saving
# + plot model weights
# + plot data before and after classification
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
mini_batch = 5
learn_rate = 0.1
architecture = [feature_dim, 2]
network = network.Network(architecture)
# Train the network
network.default_weight_initializer()
eval, train_loss, list_ws = network.SGD(train_data, epochs, mini_batch, learn_rate,
                                        test_data=test_data, is_list_weights=True)
print("mF1 on test set: {0}".format(eval[-1]))
print("MSE loss: {0}".format(train_loss[-1]))
print("Network optimal weights:")
print(network.weights[-1])
###
# Visualize the optimized network
###
net_viz = viz.NetworkVisualiser(network)
# visualize network classification decisions
net_viz.plot_decision_boundaries(network, train_data, title="Decision surface")
net_viz.plot_neurons_cost_surface(network, train_data, title="The MSE error surface")
net_viz.plot_network_optimisation(network, train_data, list_ws, title="The MSE cost optimization")




# file_net = "./data/experiment/toy/toy_epo_{0}_btch_{1}_lr_{2}". \
#     format(epochs, mini_batch, learn_rate)
# net2.save(file_net)
# net2 = network.load(file_net)

## 2.MFoM network with Sigmoid outputs
# epochs = 100
# net = mif_network.MifNetwork(architecture, alpha=5., beta=0)
# eval, loss = net.SGD(train_data, epochs, mini_batch, learn_rate, test_data=test_data)