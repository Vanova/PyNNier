from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mayavi import mlab


def show_line(x, y):
    plt.plot(x, y)
    plt.show()


def show_scatter(x, y):
    plt.scatter(x, y)
    plt.show()

def show_microF1():
    pass

def microF1(tp, fp, fn):
    return 2. * tp / (fp + 2.*tp + fn)



