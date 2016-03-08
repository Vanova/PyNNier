from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from utils import toy_loader
from ann import network

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


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_2d(ax, x, y):
    id_lbl = (y * [1, 2]).sum(axis=1)
    colors = COLORS.take(id_lbl)
    for c, m in zip(COLORS, MARKERS):
        xr = x[np.where(colors == c)]
        if len(xr):
            ax.scatter(xr[:, 0], xr[:, 1], color=c, marker=m)


def plot_3d(ax, x, y):
    id_lbl = (y * [1, 2]).sum(axis=1)
    colors = COLORS.take(id_lbl)
    for c, m in zip(COLORS, MARKERS):
        xr = x[np.where(colors == c)]
        if len(xr):
            ax.scatter(xr[:, 0], xr[:, 1], xr[:, 2], color=c, marker=m)


def plot_toy(train_data, test_data, dev_data, feature_dim):
    if feature_dim == 2:
        tr_x, tr_y = toy_loader.plot_format(train_data)
        tst_x, tst_y = toy_loader.plot_format(test_data)
        dev_x, dev_y = toy_loader.plot_format(dev_data)
        # prepare plot window
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='row', sharey='row', figsize=(10, 6))
        plt.subplots_adjust(bottom=.15)
        # plot training set
        plot_2d(ax1, tr_x, tr_y)
        ax1.set_title('training set')
        # plot test set
        plot_2d(ax2, tst_x, tst_y)
        ax2.set_title('test set')
        ax2.set_xlim(left=0, auto=True)
        ax2.set_ylim(bottom=0, auto=True)
        # plot dev
        plot_2d(ax3, dev_x, dev_y)
        ax3.set_title('development set')
        ax3.set_xlim(left=0, auto=True)
        ax3.set_ylim(bottom=0, auto=True)
        plt.show()
    if feature_dim == 3:
        tr_x, tr_y = toy_loader.plot_format(train_data)
        tst_x, tst_y = toy_loader.plot_format(test_data)
        dev_x, dev_y = toy_loader.plot_format(dev_data)

        fig = plt.figure(figsize=(12, 5))
        # plot training set
        ax1 = fig.add_subplot(131, projection='3d')
        plot_3d(ax1, tr_x, tr_y)
        ax1.set_title('training set')
        # plot test set
        ax2 = fig.add_subplot(132, projection='3d')
        plot_3d(ax2, tst_x, tst_y)
        ax2.set_title('test set')
        # plot dev
        ax3 = fig.add_subplot(133, projection='3d')
        plot_3d(ax3, dev_x, dev_y)
        ax3.set_title('development set')
        plt.show()
    if feature_dim > 3:
        print("Not implemented...")


def data_stats(data):
    feats,t = toy_loader.plot_format(data)
    print("Mean stats: {0}".format(np.mean(feats,axis=0)))
    print("Std. stats: {0}".format(np.std(feats,axis=0)))
    print("Min: {0}".format(np.min(feats,axis=0)))
    print("Max: {0}".format(np.max(feats,axis=0)))
    plt.figure(figsize=(3, 4))
    plt.boxplot(feats)
    plt.show()




#def plot_in_out_surface():


#def plot_error_surface():
    """Graph loss(W)"""


# TODO:
# - implement NN saving
# 2. plot model weights
# 2.1. plot data before and after classification
# - plot toy 2D data
# - plot toy 3D data
# 3. plot loss surface
# 4. simple architecture [2, 2] or [3, 3]
# 5. make video of parameters optimisation

feature_dim = 2
train_data, dev_data, test_data = toy_loader.load_data(n_tr=250, n_dev=50, n_tst=50,
                                                       n_features=feature_dim)
plot_toy(train_data, test_data, dev_data, feature_dim)
#data_stats(train_data)

# simple toy experiment
# 1. MSE loss and Sigmoid outputs
epochs = 10
mini_batch = 250
learn_rate = 1.0
architecture = [feature_dim,50, 2]
file_net = "./data/experiment/toy/toy_epo_{0}_btch_{1}_lr_{2}".\
        format(epochs, mini_batch, learn_rate)
net2 = network.Network(architecture)
eval, loss = net2.SGD(train_data, epochs, mini_batch, learn_rate, test_data=test_data)
net2.save(file_net)
net2 = network.load(file_net)
#print("mF1 on test set: {0}".format(eval))
#print("MSE loss: {0}".format(loss))


# plot nnet curve surface
fig = plt.figure()
ax = fig.gca(projection='3d')
gridN = 100
nx, ny = (gridN, gridN)
x = np.linspace(0, 70, nx)
y = np.linspace(0, 70, ny)
X, Y = np.meshgrid(x, y)

x = X.reshape((gridN*gridN,1))
y = Y.reshape((gridN*gridN,1))
gdata = [np.array([a.tolist(), b.tolist()]) for a,b in zip(x, y)]

#out = [(np.argmax(net2.feedforward(x)), np.argmax(y))
#                       for (x, y) in test_data]

out = [net2.feedforward(a) for a in gdata]
out = toy_loader.plot_format_no_ticks(out)

out1 = [a[1] for a in out]
out1 = np.array(out1)
out1 = out1.reshape((gridN, gridN))

R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, out1, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()













# print('The data was generated from (random_state=%d):' % 1)
# print('Class', 'P(C)', 'P(w0|C)', 'P(w1|C)', sep='\t')
# for k, p, p_w in zip(['red', 'blue', 'yellow'], p_c, p_w_c.T):
#     print('%s\t%0.2f\t%0.2f\t%0.2f' % (k, p, p_w[0], p_w[1]))
