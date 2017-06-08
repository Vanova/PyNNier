import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import mfom.visual.plot_styles as plts


def OneVsAllFunction(dim=3, fname=None):
    if dim == 3:
        gridN = 100
        nx, ny = (gridN, gridN)
        s1 = np.linspace(0, 1.0, nx)
        s2 = np.linspace(0, 1.0, ny)
        S1, S2 = np.meshgrid(s1, s2)
        x = S1.reshape((gridN * gridN, 1))
        y = S2.reshape((gridN * gridN, 1))

        # calc strat function
        den = np.exp(x) + np.exp(y)
        normX = np.exp(x) / den
        normY = np.exp(y) / den

        ebeta = math.exp(-0)
        tmp = 1.0 / normX - 1.0
        dk1 = np.log(tmp)
        l1 = 1.0 / (1.0 + np.power(tmp, -10.0) * ebeta)
        tmp = 1.0 / normY - 1.0
        dk2 = np.log(tmp)
        l2 = 1.0 / (1.0 + np.power(tmp, -10.0) * ebeta)

        dk1 = dk1.reshape((gridN, gridN))
        l1 = l1.reshape((gridN, gridN))
        dk2 = dk2.reshape((gridN, gridN))
        l2 = l2.reshape((gridN, gridN))

        # plot
        # plt.clf()  # Clear the current figure (prevents multiple labels)
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.set_title("One vs all strategy: $d_1$", fontdict=plts.title_font)
        surf = ax.plot_surface(S1, S2, dk1, rstride=3, cstride=3, alpha=0.8, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel('$\sigma_1$', fontdict=plts.label_font)
        ax.set_ylabel('$\sigma_2$', fontdict=plts.label_font)
        ax.set_zlabel('$d_1$', fontdict=plts.label_font)
        axisFormat(ax)
        cb = fig.colorbar(surf, shrink=0.5, aspect=5)
        cb.ax.tick_params(labelsize=8)

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.set_title("One vs all strategy: $d_2$", fontdict=plts.title_font)
        surf = ax.plot_surface(S1, S2, dk2, rstride=3, cstride=3, alpha=0.8, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel('$\sigma_1$', fontdict=plts.label_font)
        ax.set_ylabel('$\sigma_2$', fontdict=plts.label_font)
        ax.set_zlabel('$d_2$', fontdict=plts.label_font)
        axisFormat(ax)
        cb = fig.colorbar(surf, shrink=0.5, aspect=5)
        cb.ax.tick_params(labelsize=8)

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.set_title("One vs all strategy: $l_1$", fontdict=plts.title_font)
        surf = ax.plot_surface(S1, S2, l1, rstride=3, cstride=3, alpha=0.8, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel('$\sigma_1$', fontdict=plts.label_font)
        ax.set_ylabel('$\sigma_2$', fontdict=plts.label_font)
        ax.set_zlabel('$l_1$', fontdict=plts.label_font)
        axisFormat(ax)
        cb = fig.colorbar(surf, shrink=0.5, aspect=5)
        cb.ax.tick_params(labelsize=8)

        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.set_title("One vs all strategy: $l_2$", fontdict=plts.title_font)
        surf = ax.plot_surface(S1, S2, l2, rstride=3, cstride=3, alpha=0.8, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel('$\sigma_1$', fontdict=plts.label_font)
        ax.set_ylabel('$\sigma_2$', fontdict=plts.label_font)
        ax.set_zlabel('$l_2$', fontdict=plts.label_font)
        axisFormat(ax)
        cb = fig.colorbar(surf, shrink=0.5, aspect=5)
        cb.ax.tick_params(labelsize=8)
        plt.show()

        if fname != None:
            fig.savefig(fname)

    if dim > 3:
        t = np.arange(0., 5., 0.2)
        # red dashes, blue squares and green triangles
        plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
        plt.show()


def axisFormat(axis):
    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(7)
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(7)
    for tick in axis.zaxis.get_major_ticks():
        tick.label.set_fontsize(7)

        # ax.set_title("One vs all strategy: d_2")
        # for i in xrange(len(data_list)):
        # ax = fig.add_subplot(1, len(data_list), i+1, projection='3d')
        # ax.set_title(SET_TITLES[i])
        #     tr_x, tr_y = toy_loader.plot_format(data_list[i])
        #     plot_3d(ax, tr_x, tr_y)
        # plt.show()


# Plot value of d_k based on the outputs of sigma
# Strategy: one vs all
fn = "data/experiment/strategy/strategy_one_vs_all.png"
OneVsAllFunction(dim=3, fname=fn)
