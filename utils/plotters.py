from mpl_toolkits.mplot3d import axes3d
import copy
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import plot_styles as plts
import seaborn as sns
from matplotlib.colors import colorConverter, ListedColormap
import toy_loader

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

POINT_STYLES = np.array([('#FF3333', '^'),
                         ('#0198E1', 'v'),
                         ('#4DBD33', 'o')])


def plot_2d_scatter(ax, x, y):
    # take data points by the color
    id_lbl = (y * [1, 2]).sum(axis=1)
    colors = COLORS.take(id_lbl)
    # POINT_STYLES.take([1, 2, 0, 2, 1, 1], axis=0)
    for c, m in zip(COLORS, MARKERS):
        xr = x[np.where(colors == c)]  # take class by color
        if len(xr):
            ax.scatter(xr[:, 0], xr[:, 1], color=c, marker=m, edgecolor='black')


def plot_hyperplane(ax, clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    ax.plot(xx, yy, linestyle, label=label)


def plot_3d_scatter(ax, x, y):
    id_lbl = (y * [1, 2]).sum(axis=1)
    colors = COLORS.take(id_lbl)
    for c, m in zip(COLORS, MARKERS):
        xr = x[np.where(colors == c)]
        if len(xr):
            ax.scatter(xr[:, 0], xr[:, 1], xr[:, 2], color=c, marker=m, edgecolor='b')


def contour_view(ax, x, y, z, xlim, ylim, labels, grid=True):
    cntr = ax.contourf(x, y, z, cmap=cm.coolwarm, alpha=0.5)
    ax.set_xlabel(labels[0], fontdict=plts.label_font)
    ax.set_ylabel(labels[1], fontdict=plts.label_font)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if grid:
        ax.grid()
    return cntr


def surface_view(ax, x, y, z, xlim, ylim, labels):
    surf = ax.plot_surface(x, y, z,
                           rstride=1, cstride=1, linewidth=0,
                           alpha=0.5, cmap=cm.coolwarm, antialiased=True, zorder=0)
    ax.set_xlabel(labels[0], fontdict=plts.label_font)
    ax.set_ylabel(labels[1], fontdict=plts.label_font)
    ax.set_zlabel(labels[2], fontdict=plts.label_font)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return surf


def show_curves(y, legend=None, labels=None, title=None):
    """
    Plot curve, starting from x = 0
    """
    for it in y:
        plt.plot(it)
    if legend:
        plt.legend(legend, loc='upper left')
    if labels:
        plt.xlabel(labels[0], fontdict=plts.label_font)
        plt.ylabel(labels[1], fontdict=plts.label_font)
    if title:
        plt.title(title, fontdict=plts.title_font)
    plt.grid()
    plt.show()


def show_scatter(x, y):
    plt.scatter(x, y)
    plt.show()


class NetworkVisualiser:
    _color_bar_settings = {'ticks': np.linspace(0, 1, 9), 'format': '%.1f'}

    def __init__(self, network):
        self.optimal_weight = copy.deepcopy(network.weights)
        self.n_grid_dots = 30

    def plot_neurons_cost_surface(self, network, data, xlim, ylim, fun_name="$E$", title="Cost function surface"):
        # TODO Data vectorisation
        smps, labs = map(list, zip(*data))
        smps_vec = np.array([s.flatten() for s in smps]).T
        labs_vec = np.array([s.flatten() for s in labs]).T

        # Visualize the weights of the last layer
        fig = plt.figure(figsize=plt.figaspect(0.5))
        # Take the last network layer
        id_layer = len(network.weights) - 1
        neurons = network.weights[id_layer].shape[0]
        for i in xrange(neurons):
            ws1, ws2, cost_ws = self._cost_surface_grid(xlim, ylim, network, smps_vec, labs_vec, id_layer, i)
            ax = fig.add_subplot(2, neurons, i + 1, projection='3d')
            axis_labs = ["$w_{%d1}$" % (i + 1), "$w_{%d2}$" % (i + 1), fun_name]
            surface_view(ax, ws1, ws2, cost_ws, xlim, ylim, axis_labs)
            # surface projection
            ax = fig.add_subplot(2, neurons, (i + 1) + neurons)
            cntr = contour_view(ax, ws1, ws2, cost_ws, xlim, ylim, axis_labs)
            fig.colorbar(cntr, **self._color_bar_settings)
        plt.suptitle(title, fontdict=plts.title_font)
        plt.show()

    def plot_decision_boundaries(self, network, data, xlim, ylim, title="Network decision boundary"):
        """
        Plot decision boundary of every neuron
        """
        fig = plt.figure(figsize=plt.figaspect(0.5))
        # Initialize and fill the classification plane
        neurons = network.weights[-1].shape[0]
        for n in xrange(neurons):
            xx, yy, class_score = self._decision_grid(xlim, ylim, network, n)
            # Decision plane
            ax = fig.add_subplot(1, neurons, n + 1)
            axis_labs = ["$x$", "$y$"]
            cntr = contour_view(ax, xx, yy, class_score, xlim, ylim, axis_labs, grid=False)
            # Data scatter plot
            data_x, data_y = toy_loader.plot_format(data)
            plot_2d_scatter(ax, data_x, data_y)
            fig.colorbar(cntr, **self._color_bar_settings)
        plt.suptitle(title, fontdict=plts.title_font)
        plt.show()

    def plot_network_optimisation(self, network, data, xlim, ylim, opt_weights=None, fun_name="$E$",
                                  title="Cost function surface"):
        # Data vectorisation
        smps, labs = map(list, zip(*data))
        xvec = np.array([s.flatten() for s in smps]).T
        yvec = np.array([s.flatten() for s in labs]).T
        # Visualize the weights of the last layer
        fig = plt.figure(figsize=plt.figaspect(0.5))
        id_layer = len(network.weights) - 1
        neurons = network.weights[id_layer].shape[0]
        for i in xrange(neurons):
            ws1, ws2, cost_ws = self._cost_surface_grid(xlim, ylim, network, xvec, yvec, id_layer, i)
            # plot overview of cost function
            ax = fig.add_subplot(2, neurons, i + 1, projection='3d')
            axis_labs = ["$w_{%d1}$" % (i + 1), "$w_{%d2}$" % (i + 1), fun_name]
            surface_view(ax, ws1, ws2, cost_ws, xlim, ylim, axis_labs)

            # optimisation weights on the 3D surface
            for ww in opt_weights:
                network.weights = copy.deepcopy(ww)
                opt_c = network.cost_value(network.feedforward(xvec), yvec)
                ax.scatter(ww[id_layer][i, 0], ww[id_layer][i, 1], opt_c, c='g', s=50, edgecolor='g', marker='.',
                           zorder=2)
            network.weights = copy.deepcopy(opt_weights[-1])
            opt_c = network.cost_value(network.feedforward(xvec), yvec)
            ax.scatter(opt_weights[-1][id_layer][i, 0], opt_weights[-1][id_layer][i, 1], opt_c,
                       c='red', s=100, marker='*', zorder=2)

            # surface projection of cost function
            ax = fig.add_subplot(2, neurons, (i + 1) + neurons)
            cntr = contour_view(ax, ws1, ws2, cost_ws, xlim, ylim, axis_labs)
            # weights on 2D projection
            for ww in opt_weights:
                ax.scatter(ww[id_layer][i, 0], ww[id_layer][i, 1], c='g', s=50, edgecolor='g', marker='.')
            ax.scatter(opt_weights[-1][id_layer][i, 0], opt_weights[-1][id_layer][i, 1], c='red', s=100, marker='*')

            fig.colorbar(cntr, **self._color_bar_settings)
        plt.suptitle(title, fontdict=plts.title_font)
        plt.show()
        # TODO 3D weights case

    def _decision_grid(self, xlim, ylim, network, neuron_id):
        xs1 = np.linspace(*xlim, num=self.n_grid_dots)
        xs2 = np.linspace(*ylim, num=self.n_grid_dots)
        xx, yy = np.meshgrid(xs1, xs2)
        # Initialize and fill the classification plane
        class_surface = np.zeros((self.n_grid_dots, self.n_grid_dots))
        for i in range(self.n_grid_dots):
            xy = np.array([xx[i], yy[i]])
            pred = network.feedforward(xy)
            class_surface[i] = pred[neuron_id]
        return xx, yy, class_surface

    def _cost_surface_grid(self, xlim, ylim, network, data, labels, layer, neuron_id):
        """Define a vector of weights for which we want to plot the cost."""
        w1 = np.linspace(*xlim, num=self.n_grid_dots)  # Weight 1
        w2 = np.linspace(*ylim, num=self.n_grid_dots)  # Weight 2
        ws1, ws2 = np.meshgrid(w1, w2)  # Generate grid
        cost_ws = np.zeros((self.n_grid_dots, self.n_grid_dots))  # Initialize cost matrix
        # Fill the cost matrix for each combination of weights
        for i in range(self.n_grid_dots):
            for j in range(self.n_grid_dots):
                network.weights[layer][neuron_id] = np.array([ws1[i, j], ws2[i, j]])
                cost_ws[i, j] = network.cost_value(network.feedforward(data), labels)
        # TODO copy back optimal weights to network
        network.weights = copy.deepcopy(self.optimal_weight)
        return ws1, ws2, cost_ws
