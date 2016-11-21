from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patheffects import withStroke


def sigma(x):
    return 1. / (1. + np.exp(-x))


def softmax(*args):
    sm = []
    for x in args:
        sm.append(np.exp(x) / np.exp(args))
    return sm


def field3Dview(ax, X, Y, Z, U, V, W, **options):
    ar = 0.1
    if options.get("arrow"):
        ar = options.get("arrow")
    if options.get("labels"):
        l = options.get("labels")
        ax.set_xlabel(l[0], fontsize=14)
        ax.set_ylabel(l[1], fontsize=14)
        ax.set_zlabel(l[2], fontsize=14)
    if options.get("title"):
        ax.set_title(title=options.get("title"))
    ax.quiver(X, Y, Z, U, V, W,
              length=ar,
              cmap=cm.coolwarm)
    ax.view_init(elev=18, azim=30)
    ax.dist = 8


def field2Dview(ax, X, Y, U, V, **options):
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
field2Dview(ax, X, Y, SX, SY, gradient=True,
            labels=["$z_1$", "$z_2$"],
            title="Sigmoid vector function: $(\sigma_1(z_1), \sigma_2(z_2))$")

### discriminative function ``d_k'' 2D vector field
d1 = -SX + np.log(np.exp(SY) + 1e-6)
d2 = -SY + np.log(np.exp(SX) + 1e-6)
ax = fig.add_subplot(N, M, 2)
field2Dview(ax, X, Y, d1, d2, gradient=True,
            labels=["$z_1$", "$z_2$"],
            title="Misclassification $D($z$)$ : $(d_1($z$)$, $d_2($z$))$")

### class loss function ``l_k(d_k)'' 2D vector field
Lu = sigma(d1)
Lv = sigma(d2)
ax = fig.add_subplot(N, M, 3)
field2Dview(ax, d1, d2, Lu, Lv, gradient=True,
            labels=["$d_1$", "$d_2$"],
            title="Class loss $L(D)$ : $(l_1(d_1), l_2(d_2))$")

### plot discriminative function ``d_k'' 2D vector field
# "units-vs-zeros" case (1, 1) label
# NOTE: it is exactly as Sigmoid DNN, BUT the gradient is changing
# UNSTABLE CASE!!!
d1 = -SX  # + np.log(np.exp(SY)+1e-6)
d2 = -SY  # + np.log(np.exp(SX)+1e-6)
ax = fig.add_subplot(N, M, 4)
field2Dview(ax, X, Y, d1, d2, gradient=True,
            labels=["$z_1$", "$z_2$"],
            title="Misclassification \"u-v-z\" $D($z$)$ : $(d_1($z$)$, $d_2($z$))$")
plt.show()

###
# strategy function, 3D vector field
###
M, N = 2, 3
fig = plt.figure()
# some activation signal
X, Y, Z = np.meshgrid(np.arange(-5, 6, 2),
                      np.arange(-5, 6, 2),
                      np.arange(-5, 6, 2))
### Sigmoid outputs, 3D field
SX = sigma(X)
SY = sigma(Y)
SZ = sigma(Z)
ax = fig.add_subplot(M, N, 1, projection='3d')
field3Dview(ax, X, Y, Z, SX, SY, SZ, arrow=0.9)

### Pure MFoM: "one-vs-others" strategy, discriminative function d_k
U = -SX + np.log(0.5 * (np.exp(SY) + np.exp(SZ)))
V = -SY + np.log(0.5 * (np.exp(SX) + np.exp(SZ)))
W = -SZ + np.log(0.5 * (np.exp(SX) + np.exp(SY)))
ax = fig.add_subplot(M, N, 2, projection='3d')
field3Dview(ax, SX, SY, SZ, U, V, W, arrow=0.1)

# 1, 0, 0
U = -SX + np.log(0.5 * (np.exp(SY) + np.exp(SZ)))
V = -SY + np.log((np.exp(SX)))
W = -SZ + np.log((np.exp(SX)))
ax = fig.add_subplot(M, N, 3, projection='3d')
field3Dview(ax, SX, SY, SZ, U, V, W, arrow=0.1)

# 0, 1, 0
U = -SX + np.log(np.exp(SY))
V = -SY + np.log(0.5 * (np.exp(SX) + np.exp(SZ)))
W = -SZ + np.log(np.exp(SY))
ax = fig.add_subplot(M, N, 4, projection='3d')
field3Dview(ax, SX, SY, SZ, U, V, W, arrow=0.1)

# 0, 0, 1
U = -SX + np.log(np.exp(SZ))
V = -SY + np.log(np.exp(SZ))
W = -SZ + np.log(0.5 * (np.exp(SX) + np.exp(SY)))
ax = fig.add_subplot(M, N, 5, projection='3d')
field3Dview(ax, SX, SY, SZ, U, V, W, arrow=0.1)

# 1, 1, 0
U = -SX + np.log(np.exp(SZ))
V = -SY + np.log(np.exp(SZ))
W = -SZ + np.log(0.5 * (np.exp(SX) + np.exp(SY)))
ax = fig.add_subplot(M, N, 6, projection='3d')
field3Dview(ax, SX, SY, SZ, U, V, W, arrow=0.1)
plt.show()
