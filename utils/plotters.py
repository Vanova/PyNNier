from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def show_line(x, y):
    plt.plot(x, y)
    plt.show()


def show_scatter(x, y):
    plt.scatter(x, y)
    plt.show()


class NetworkVisualiser():
    def __init__(self):
        pass

    def get_cost_surface(self, w1_low, w1_high, w2_low, w2_high, nb_of_ws, cost_func):
        """Define a vector of weights for which we want to plot the cost."""
        w1 = np.linspace(w1_low, w1_high, num=nb_of_ws)  # Weight 1
        w2 = np.linspace(w2_low, w2_high, num=nb_of_ws)  # Weight 2
        ws1, ws2 = np.meshgrid(w1, w2)  # Generate grid
        cost_ws = np.zeros((nb_of_ws, nb_of_ws))  # Initialize cost matrix
        # Fill the cost matrix for each combination of weights
        for i in range(nb_of_ws):
            for j in range(nb_of_ws):
                cost_ws[i, j] = cost_func(ws1[i, j], ws2[i, j])
        return ws1, ws2, cost_ws

    def plot_error_surface(self, network):
        pass

    def plot_network_optimisation(self, network, data, labels):

        pass

    def plot_surface(self, ax, ws1, ws2, cost_ws):
        """Plot the cost in function of the weights."""
        surf = ax.contourf(ws1, ws2, cost_ws, levels=np.logspace(-0.2, 8, 30), cmap=cm.pink, norm=LogNorm())
        ax.set_xlabel('$w_{in}$', fontsize=15)
        ax.set_ylabel('$w_{rec}$', fontsize=15)
        return surf

    def plot_optimisation(self, ls_of_ws, cost_func):
        """Plot the optimisation iterations on the cost surface."""
        ws1, ws2 = zip(*ls_of_ws)
        # Plot figures
        fig = plt.figure(figsize=(10, 4))

        # Plot overview of cost function
        ax_1 = fig.add_subplot(1, 2, 1)
        ws1_1, ws2_1, cost_ws_1 = self.get_cost_surface(-3, 3, -3, 3, 100, cost_func)
        surf_1 = self.plot_surface(ax_1, ws1_1, ws2_1, cost_ws_1 + 1)
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
