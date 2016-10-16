# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter, ListedColormap
import toy_loader


def show_curve(y, labels, title):
    """
    Plot curve, starting from x = 0
    """
    plt.plot(y, 'b-')
    plt.xlabel(labels[0], fontsize=15)
    plt.ylabel(labels[1], fontsize=15)
    plt.title(title)
    plt.grid()
    plt.show()


def show_scatter(x, y):
    plt.scatter(x, y)
    plt.show()


class NetworkVisualiser():
    # Define points to annotate (wx, wRec, color)
    data_points = [(2, 1, 'r'), (1, 2, 'b'), (1, -2, 'g'), (1, 0, 'c'), (1, 0.5, 'm'), (1, -0.5, 'y')]

    def plot_neurons_cost_surface(self, network, data, fun_name="$E$", title="Cost function surface"):
        # Data vectorisation
        x, y = map(list, zip(*data))
        xvec = np.array([s.flatten() for s in x]).T
        yvec = np.array([s.flatten() for s in y]).T
        # visualize the weights of the last layer
        fig = plt.figure(figsize=plt.figaspect(0.5))
        id_layer = len(network.weights) - 1
        neurons = network.weights[id_layer].shape[0]
        for i in xrange(neurons):
            ax = fig.add_subplot(2, neurons, i + 1, projection='3d')
            ws1, ws2, cost_ws = self._cost_surface_grid(-3, 3, -3, 3, 100, network, xvec, yvec, id_layer, i)
            labs = ["$w_{%d1}$" % (i + 1), "$w_{%d2}$" % (i + 1), fun_name]
            self._surface_view(ax, ws1, ws2, cost_ws, labs)
            # surface projection
            ax = fig.add_subplot(2, neurons, (i + 1) + neurons)
            cntr = self._contour_view(ax, ws1, ws2, cost_ws, labs)
            fig.colorbar(cntr, ticks=np.linspace(0, 1, 9))
        plt.suptitle(title, fontsize=15)
        plt.show()
        # TODO 3D weights case

    def plot_decision_boundaries(self, network, data, fun_name="$E$", title="Network decision boundary"):
        """
        Plot decision boundary of every neuron
        """
        id_neuron = 0
        nb_of_xs = 100
        xs1 = np.linspace(-3, 3, num=nb_of_xs)
        xs2 = np.linspace(-3, 3, num=nb_of_xs)
        xx, yy = np.meshgrid(xs1, xs2)  # create the grid
        # Initialize and fill the classification plane
        classification_plane = np.zeros((nb_of_xs, nb_of_xs))
        # TODO vectorise (xx,yy)
        for i in range(nb_of_xs):
            for j in range(nb_of_xs):
                pred = network.feedforward(np.asarray([[xx[i, j]], [yy[i, j]]]))
                # TODO plot for one neuron
                classification_plane[i, j] = pred[id_neuron, 0]
        # Create a color map to show the classification colors of each grid point
        # cmap = ListedColormap([
        #     colorConverter.to_rgba('b', alpha=0.30),
        #     colorConverter.to_rgba('r', alpha=0.30)])

        # Plot the classification plane with decision boundary and input samples
        plt.contourf(xx, yy, classification_plane, cmap=cm.coolwarm)
        # Plot both classes on the x1, x2 plane
        # TODO plot data scatter
        # plt.plot(x_red[:, 0], x_red[:, 1], 'ro', label='class red')
        # plt.plot(x_blue[:, 0], x_blue[:, 1], 'bo', label='class blue')
        # plt.grid()
        # plt.legend(loc=1)
        # plt.xlabel('$x_1$', fontsize=15)
        # plt.ylabel('$x_2$', fontsize=15)
        # plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.title(title)
        plt.show()

    def plot_network_optimisation(self, network, list_of_ws, data, labels):
        """Plot the optimisation iterations on the cost surface."""
        ws1, ws2 = zip(*list_of_ws)
        # plot figures
        fig = plt.figure(figsize=(5, 4))
        # plot overview of cost function
        ax_1 = fig.add_subplot(1, 1, 1, projection='3d')
        ws1_1, ws2_1, cost_ws_1 = self._cost_surface_grid(-3, 3, -3, 3, 100, network, data, labels)
        surf_1 = self._surface_view(ax_1, ws1_1, ws2_1, cost_ws_1 + 1)
        ax_1.plot(ws1, ws2, 'b.')
        ax_1.set_xlim([-3, 3])
        ax_1.set_ylim([-3, 3])
        # Show the colorbar
        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.12, 0.03, 0.78])
        cbar = fig.colorbar(surf_1, ticks=np.logspace(0, 8, 9), cax=cax)
        cbar.ax.set_ylabel('$\\xi$', fontsize=15)
        cbar.set_ticklabels(['{:.0e}'.format(i) for i in np.logspace(0, 8, 9)])
        plt.suptitle('Cost surface', fontsize=15)
        plt.show()

    def plot_gradient_over_time(self, points, get_gradient_func, data, labels):
        """Plot the gradients of the annotated point and how the evolve over time."""
        fig = plt.figure(figsize=(6.5, 4))
        ax = plt.subplot(111)
        # Plot points
        for wx, wRec, c in points:
            grad_over_time = get_gradient_func(wx, wRec, data, labels)
            x = np.arange(-grad_over_time.shape[1] + 1, 1, 1)
            plt.plot(x, np.sum(grad_over_time, axis=0), c + '-', label='({0}, {1})'.format(wx, wRec), linewidth=1,
                     markersize=8)
        plt.xlim(0, -grad_over_time.shape[1] + 1)
        # Set up plot axis
        plt.xticks(x)
        plt.yscale('symlog')
        plt.yticks([10 ** 8, 10 ** 6, 10 ** 4, 10 ** 2, 0, -10 ** 2, -10 ** 4, -10 ** 6, -10 ** 8])
        plt.xlabel('timestep k', fontsize=12)
        plt.ylabel('$\\frac{\\partial \\xi}{\\partial S_{k}}$', fontsize=20, rotation=0)
        plt.grid()
        plt.title('Unstability of gradient in backward propagation.\n(backpropagate from left to right)')
        # Set legend
        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, numpoints=1)
        leg.set_title('$(w_x, w_{rec})$', prop={'size': 15})

    def _cost_surface_grid(self, w1_low, w1_high, w2_low, w2_high, nb_of_ws, network, data, labels, layer, neuron):
        """Define a vector of weights for which we want to plot the cost."""
        w1 = np.linspace(w1_low, w1_high, num=nb_of_ws)  # Weight 1
        w2 = np.linspace(w2_low, w2_high, num=nb_of_ws)  # Weight 2
        ws1, ws2 = np.meshgrid(w1, w2)  # Generate grid
        cost_ws = np.zeros((nb_of_ws, nb_of_ws))  # Initialize cost matrix
        # Fill the cost matrix for each combination of weights
        for i in range(nb_of_ws):
            for j in range(nb_of_ws):
                network.weights[layer][neuron] = np.array([ws1[i, j], ws2[i, j]])
                cost_ws[i, j] = network.cost_value(network.feedforward(data), labels)
        return ws1, ws2, cost_ws

    # def _get_cost_surface(self, w1_low, w1_high, w2_low, w2_high, nb_of_ws, network, data, labels):
    #     """Define a vector of weights for which we want to plot the cost."""
    #     w1 = np.linspace(w1_low, w1_high, num=nb_of_ws)  # Weight 1
    #     w2 = np.linspace(w2_low, w2_high, num=nb_of_ws)  # Weight 2
    #     ws1, ws2 = np.meshgrid(w1, w2)  # Generate grid
    #     cost_ws = np.zeros((nb_of_ws, nb_of_ws))  # Initialize cost matrix
    #     # Fill the cost matrix for each combination of weights
    #     for i in range(nb_of_ws):
    #         for j in range(nb_of_ws):
    #             network.W = [ws1[i, j], ws2[i, j]]
    #             cost_ws[i, j] = network.cost_value(network.feedforward(data)[:, -1], labels)
    #     return ws1, ws2, cost_ws

    def _surface_view(self, ax, ws1, ws2, cost_ws, labels):
        """Plot the cost in function of the weights."""
        surf = ax.plot_surface(ws1, ws2, cost_ws,
                               rstride=1, cstride=1, linewidth=0,
                               alpha=1, cmap=cm.coolwarm, antialiased=False)
        ax.set_xlabel(labels[0], fontsize=15)
        ax.set_ylabel(labels[1], fontsize=15)
        ax.set_zlabel(labels[2], fontsize=15)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        return surf

    def _contour_view(self, ax, ws1, ws2, cost_ws, labels):
        surf = ax.contourf(ws1, ws2, cost_ws, cmap=cm.coolwarm)
        ax.set_xlabel(labels[0], fontsize=15)
        ax.set_ylabel(labels[1], fontsize=15)
        ax.grid()
        return surf
