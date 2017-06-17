"""
Ref. to check: https://youtu.be/1P-MhIL9_7c
stream slice
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patheffects import withStroke
from matplotlib import animation


def sigma(x):
    return 1. / (1. + np.exp(-1. * x))


def field_3d_view(ax, X, Y, Z, U, V, W, **options):
    ar = 0.1
    if options.get("arrow"):
        ar = options.get("arrow")
    if options.get("labels"):
        l = options.get("labels")
        ax.set_xlabel(l[0], fontsize=14)
        ax.set_ylabel(l[1], fontsize=14)
        ax.set_zlabel(l[2], fontsize=14)
    if options.get("title"):
        ax.set_title(options.get("title"))
    Q = ax.quiver(X, Y, Z, U, V, W,
              length=ar,
              cmap=cm.coolwarm)
    ax.view_init(elev=18, azim=30)
    ax.dist = 8
    return Q, ax


def field_2d_view(ax, X, Y, U, V, **options):
    length = np.sqrt(U ** 2 + V ** 2)
    if options.get("gradient"):
        q = ax.quiver(X, Y, U, V,
                      length,
                      cmap=cm.coolwarm,
                      headlength=7)
        plt.colorbar(q)
    if options.get("contour"):
        cont = ax.contour(X, Y, length, cmap='gist_earth')
        ax.clabel(cont)
    if options.get("stream"):
        ax.streamplot(X, Y, U, V, color=length, density=0.5, cmap='gist_earth')
    if options.get("labels"):
        l = options.get("labels")
        ax.set_xlabel(l[0], fontsize=14)
        ax.set_ylabel(l[1], fontsize=14)
    if options.get("title"):
        ax.set(aspect=1, title=options.get("title"))
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    return None


###
# strategy function 2D vector field
###
X, Y = np.meshgrid(np.linspace(-10, 10, 10),
                   np.linspace(-10, 10, 10))
SX = sigma(X)
SY = sigma(Y)
### sigmoid 2D vector field
fig = plt.figure(figsize=plt.figaspect(0.5))
N, M = 2, 2
ax = fig.add_subplot(N, M, 1)
field_2d_view(ax, X, Y, SX, SY, gradient=True,
              labels=["$z_1$", "$z_2$"],
              title="Sigmoid vector function: $(\sigma_1(z_1), \sigma_2(z_2))$")

### discriminative function ``d_k'' 2D vector field
d1 = -SX + np.log(np.exp(SY) + 1e-6)
d2 = -SY + np.log(np.exp(SX) + 1e-6)
ax = fig.add_subplot(N, M, 2)
field_2d_view(ax, X, Y, d1, d2, gradient=True,
              labels=["$z_1$", "$z_2$"],
              title="Misclassification $D($z$)$ : $(d_1($z$)$, $d_2($z$))$")

### class loss function ``l_k(d_k)'' 2D vector field
Lu = sigma(d1)
Lv = sigma(d2)
ax = fig.add_subplot(N, M, 3)
field_2d_view(ax, d1, d2, Lu, Lv, gradient=True,
              labels=["$d_1$", "$d_2$"],
              title="Class loss $L(D)$ : $(l_1(d_1), l_2(d_2))$")

### plot discriminative function ``d_k'' 2D vector field
# "units-vs-zeros" case (1, 1) label
# NOTE: it is exactly as Sigmoid DNN, BUT the gradient is changing
# UNSTABLE CASE!!!
d1 = -SX  # + np.log(np.exp(SY)+1e-6)
d2 = -SY  # + np.log(np.exp(SX)+1e-6)
ax = fig.add_subplot(N, M, 4)
field_2d_view(ax, X, Y, d1, d2, gradient=True,
              labels=["$z_1$", "$z_2$"],
              title="Misclassification \"u-v-z\" $D($z$)$ : $(d_1($z$)$, $d_2($z$))$")
plt.show()

###
# strategy function, 3D vector field
###
M, N = 2, 2
# some activation signal
X, Y, Z = np.meshgrid(np.arange(-5, 6, 2),
                      np.arange(-5, 6, 2),
                      np.arange(-5, 6, 2))
### Sigmoid outputs, 3D field
SX = sigma(X)
SY = sigma(Y)
SZ = sigma(Z)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
field_3d_view(ax, X, Y, Z, SX, SY, SZ, arrow=0.9)
plt.show()

### Pure MFoM: "one-vs-others" strategy, discriminative function d_k
U = -SX + np.log(0.5 * (np.exp(SY) + np.exp(SZ)))
V = -SY + np.log(0.5 * (np.exp(SX) + np.exp(SZ)))
W = -SZ + np.log(0.5 * (np.exp(SX) + np.exp(SY)))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
field_3d_view(ax, SX, SY, SZ, U, V, W, arrow=0.1, title="MFoM: one-vs-others strategy")
plt.show()

# 1, 0, 0
U = -SX + np.log(0.5 * (np.exp(SY) + np.exp(SZ)))
V = -SY + np.log((np.exp(SX)))
W = -SZ + np.log((np.exp(SX)))
UL = sigma(U)
VL = sigma(V)
WL = sigma(W)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
field_3d_view(ax, SX, SY, SZ, U, V, W, arrow=0.1, title="MFoM: units-vs-zeros strategy, (1, 0, 0)")
plt.show()


# 1, 1, 0
U = -SX + np.log(np.exp(SZ))
V = -SY + np.log(np.exp(SZ))
W = -SZ + np.log(0.5 * (np.exp(SX) + np.exp(SY)))

UL = sigma(U)
VL = sigma(V)
WL = sigma(W)

fig = plt.figure()
# fig, ax = plt.subplots(1,1)
ax = fig.add_subplot(1, 1, 1, projection='3d')
Q, ax = field_3d_view(ax, SX, SY, SZ, U, V, W, arrow=0.1, title="MFoM: units-vs-zeros strategy, (1, 1, 0)")
plt.ion()
plt.show()

def update_quiver(num, Q, ax, X, Y, Z):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    SX = sigma(0.01 * num + X)
    SY = sigma(0.01 * num + Y)
    SZ = sigma(0.01 * num + Z)

    # U = -SX + np.log(np.exp(SZ))
    # V = -SY + np.log(np.exp(SZ))
    # W = -SZ + np.log(0.5 * (np.exp(SX) + np.exp(SY)))

    U = -SX + np.log(0.5 * (np.exp(SY) + np.exp(SZ)))
    V = -SY + np.log((np.exp(SX)))
    W = -SZ + np.log((np.exp(SX)))

    # U = -SX + np.log(0.5 * (np.exp(SY) + np.exp(SZ)))
    # V = -SY + np.log(0.5 * (np.exp(SX) + np.exp(SZ)))
    # W = -SZ + np.log(0.5 * (np.exp(SX) + np.exp(SY)))

    ax.cla()
    ax.quiver(SX, SY, SZ, U, V, W,
                  length=0.1,
                  cmap=cm.coolwarm)
    ax.view_init(elev=18, azim=30)
    ax.dist = 8

    plt.draw()
    return Q,

anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, ax, X, Y, Z),
                               interval=100, blit=False)
plt.ioff()
plt.show()


# # 0, 1, 0
# U = -SX + np.log(np.exp(SY))
# V = -SY + np.log(0.5 * (np.exp(SX) + np.exp(SZ)))
# W = -SZ + np.log(np.exp(SY))
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# field_3d_view(ax, SX, SY, SZ, U, V, W, arrow=0.1)
# plt.show()
#
# # 0, 0, 1
# U = -SX + np.log(np.exp(SZ))
# V = -SY + np.log(np.exp(SZ))
# W = -SZ + np.log(0.5 * (np.exp(SX) + np.exp(SY)))
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# field_3d_view(ax, SX, SY, SZ, U, V, W, arrow=0.1)
# plt.show()

# if __name__=='__main__':


