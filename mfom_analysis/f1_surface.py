import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(0.01, 1.01, 0.1)
y = np.arange(0.01, 1.01, 0.1)

X, Y = np.meshgrid(x, y)

def F(x, y):
    return 1. - 2. * x * y / (x + y)

Z = F(X, Y)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0.7, antialiased=False, alpha=0.7)
ax.set_xlabel('$\mathcal{\overline{L}}$', size=14)
ax.set_ylabel('$\mathbf{Y}$', size=14)
ax.set_zlabel('$E_F$', size=14)
plt.show()

fig.savefig("f1_surface.pdf", bbox_inches='tight')
