from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import math

# plot settings
ticksfont = {
    'family': 'sans-serif',  # (cursive, fantasy, monospace, serif)
    'color': 'black',  # html hex or colour name
    'weight': 'normal',  # (normal, bold, bolder, lighter)
    'size': 5,  # default value:12
}
labelfont = {
    'family': 'sans-serif',  # (cursive, fantasy, monospace, serif)
    'color': 'black',  # html hex or colour name
    'weight': 'normal',  # (normal, bold, bolder, lighter)
    'size': 8,  # default value:12
}
titlefont = {
    'family': 'serif',
    'color': 'black',
    'weight': 'bold',
    'size': 10,
}


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
        ax.set_title("One vs all strategy: d_1", fontdict=titlefont)
        surf = ax.plot_surface(S1, S2, dk1, rstride=3, cstride=3, alpha=0.8, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel('sigma_1', fontdict=labelfont)
        ax.set_ylabel('sigma_2', fontdict=labelfont)
        ax.set_zlabel('d_1', fontdict=labelfont)
        axisFormat(ax)
        cb = fig.colorbar(surf, shrink=0.5, aspect=5)
        cb.ax.tick_params(labelsize=8)

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.set_title("One vs all strategy: d_2", fontdict=titlefont)
        surf = ax.plot_surface(S1, S2, dk2, rstride=3, cstride=3, alpha=0.8, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel('sigma_1', fontdict=labelfont)
        ax.set_ylabel('sigma_2', fontdict=labelfont)
        ax.set_zlabel('d_2', fontdict=labelfont)
        axisFormat(ax)
        cb = fig.colorbar(surf, shrink=0.5, aspect=5)
        cb.ax.tick_params(labelsize=8)

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.set_title("One vs all strategy: l_1", fontdict=titlefont)
        surf = ax.plot_surface(S1, S2, l1, rstride=3, cstride=3, alpha=0.8, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel('sigma_1', fontdict=labelfont)
        ax.set_ylabel('sigma_2', fontdict=labelfont)
        ax.set_zlabel('l_1', fontdict=labelfont)
        axisFormat(ax)
        cb = fig.colorbar(surf, shrink=0.5, aspect=5)
        cb.ax.tick_params(labelsize=8)

        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.set_title("One vs all strategy: l_2", fontdict=titlefont)
        surf = ax.plot_surface(S1, S2, l2, rstride=3, cstride=3, alpha=0.8, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel('sigma_1', fontdict=labelfont)
        ax.set_ylabel('sigma_2', fontdict=labelfont)
        ax.set_zlabel('l_2', fontdict=labelfont)
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


#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# def simData():
# # this function is called as the argument for
# # the simPoints function. This function contains
# # (or defines) and iterator---a device that computes
# # a value, passes it back to the main program, and then
# # returns to exactly where it left off in the function upon the
# # next call. I believe that one has to use this method to animate
# # a function using the matplotlib animation package.
# #
#     t_max = 10.0
#     dt = 0.05
#     x = 0.0
#     t = 0.0
#     while t < t_max:
#         x = np.sin(np.pi*t)
#         t = t + dt
#         yield x, t
#
# def simPoints(simData):
#     x, t = simData[0], simData[1]
#     time_text.set_text(time_template%(t))
#     line.set_data(t, x)
#     return line, time_text
#
# ##
# ##   set up figure for plotting:
# ##
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # I'm still unfamiliar with the following line of code:
# line, = ax.plot([], [], 'bo', ms=10)
# ax.set_ylim(-1, 1)
# ax.set_xlim(0, 10)
# ##
# time_template = 'Time = %.1f s'    # prints running simulation time
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
# ## Now call the animation package: (simData is the user function
# ## serving as the argument for simPoints):
# ani = animation.FuncAnimation(fig, simPoints, simData, blit=False,\
#      interval=10, repeat=True)
# plt.show()