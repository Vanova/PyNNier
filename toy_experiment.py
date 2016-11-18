import numpy as np
import os
import matplotlib.pyplot as plt
from utils import toy_loader
from sklearn import preprocessing
import utils.plotters as viz
from ann import network, mfom_network
from ann import matrix_network
from ann import mfom_network
import copy

np.random.seed(777)

DATA_PATH = "./data/experiment/toy/"

SET_TITLES = ['Training set',
              'Test set',
              'Dev set']


def plot_toy_data(data_list, titles, feature_dim):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    if feature_dim == 2:
        for i, data in enumerate(data_list):
            ax = fig.add_subplot(1, len(data_list), i + 1)
            ax.set_title(titles[i])
            data_x, data_y = toy_loader.plot_format(data)
            viz.plot_2d_scatter(ax, data_x, data_y)
        plt.show()
    if feature_dim == 3:
        for i, data in enumerate(data_list):
            ax = fig.add_subplot(1, len(data_list), i + 1, projection='3d')
            ax.set_title(titles[i])
            data_x, data_y = toy_loader.plot_format(data)
            viz.plot_3d_scatter(ax, data_x, data_y)
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
nclass = 2
train_data, dev_data, test_data = toy_loader.load_data(n_tr=250, n_dev=50, n_tst=50,
                                                       n_features=feature_dim, n_classes=nclass,
                                                       scaler=preprocessing.StandardScaler())
plot_toy_data([train_data, dev_data, test_data], SET_TITLES, feature_dim)
data_stats(train_data)

#####
# Simple toy experiment
# 1.MSE loss and Sigmoid outputs
#####
epochs = 100
mini_batch = 5
learn_rate = 0.1
architecture = [feature_dim, nclass]
mse_network = matrix_network.MatrixNetwork(architecture)
# Train the network
eval_cost, eval_acc, tr_cost, tr_acc, list_ws = mse_network.SGD(train_data, epochs, mini_batch,
                                                                learn_rate, evaluation_data=test_data,
                                                                is_list_weights=True,
                                                                monitor_evaluation_cost=True,
                                                                monitor_evaluation_accuracy=True,
                                                                monitor_training_cost=True,
                                                                monitor_training_accuracy=True)
print("test mF1: {0}".format(eval_cost[-1]))
print("train MSE loss: {0}".format(tr_cost[-1]))
print("Network optimal weights:")
print(mse_network.weights[-1])
print("Optimal train (mse network) F1: {}".format(mse_network.accuracy(train_data)))
viz.show_curves([eval_cost, tr_cost],
                legend=["evaluation cost", "training cost"],
                labels=["# of epochs", "value"],
                title="MSE cost function")
viz.show_curves([eval_acc, tr_acc],
                legend=["evaluation acc", "training acc"],
                labels=["# of epochs", "value, %"],
                title="Micro F1 value, sigmoid scores")
##
# Visualize the optimized network
##
net_viz = viz.NetworkVisualiser(mse_network)
# visualize network classification decisions
net_viz.plot_decision_boundaries(mse_network, train_data, xlim=[-4, 4], ylim=[-4, 4],
                                 title="Decision surface")
net_viz.plot_neurons_cost_surface(mse_network, train_data, xlim=[-5, 5], ylim=[-5, 5],
                                  title="The MSE error surface")
net_viz.plot_network_optimisation(mse_network, train_data, xlim=[-5, 5], ylim=[-5, 5],
                                  opt_weights=list_ws,
                                  title="The MSE cost optimization")

file_net = os.path.join(DATA_PATH, "toy_mse_epo_{}_btch_{}_lr_{}".
                        format(epochs, mini_batch, learn_rate))
mse_network.save(file_net)
mse_network = network.load(file_net)



#####
# 2.MFoM network with Sigmoid outputs
#####
epochs = 100
mini_batch = 5
learn_rate = 0.1
architecture = [feature_dim, nclass]
mfom_network = mfom_network.MFoMNetwork(architecture, alpha=10., beta=0)
# copy MSE network weights
mfom_network.weights = copy.deepcopy(mse_network.weights)
mfom_network.biases = copy.deepcopy(mse_network.biases)
print("Optimal train (mfom network) F1: {}".format(mfom_network.accuracy(train_data, True)))
eval_cost, eval_acc, tr_cost, tr_acc, list_ws = mfom_network.SGD(train_data, epochs, mini_batch,
                                                                 learn_rate, evaluation_data=test_data,
                                                                 is_list_weights=True,
                                                                 monitor_evaluation_cost=True,
                                                                 monitor_evaluation_accuracy=True,
                                                                 monitor_training_cost=True,
                                                                 monitor_training_accuracy=True)
print("TEST smooth micro F1: {0}".format(eval_cost[-1]))
print("TRAIN smooth micro F1: {0}".format(tr_cost[-1]))
print("F1 error test (LOSS SCORES): {}".format(mfom_network.accuracy(test_data, True)))
print("F1 error test (SIGMOID): {}".format(mfom_network.accuracy(test_data)))
print("Network optimal weights:")
print(mfom_network.weights[-1])
viz.show_curves([eval_cost, tr_cost],
                legend=["evaluation cost", "training cost"],
                labels=["# of epochs", "value"],
                title="MFoM smooth F1 cost")
viz.show_curves([eval_acc, tr_acc],
                legend=["evaluation acc", "training acc"],
                labels=["# of epochs", "value, %"],
                title="MFoM micro F1 cost")

###
# Visualize the optimized network
###
net_viz = viz.NetworkVisualiser(mfom_network)
# visualize network classification decisions
net_viz.plot_decision_boundaries(mfom_network, train_data, xlim=[-4, 4], ylim=[-4, 4],
                                 title="Decision surface")
net_viz.plot_neurons_cost_surface(mfom_network, train_data, xlim=[-5, 5], ylim=[-5, 5],
                                  title="The MFoM error surface")
net_viz.plot_network_optimisation(mfom_network, train_data, xlim=[-5, 5], ylim=[-5, 5],
                                  opt_weights=list_ws,
                                  title="The MFoM cost optimization")
