from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math


def column(matrix, i):
    return [row[i] for row in matrix]


top = 1.0
bot = 0.0
N = 20

# sigmoid signal initialization
s1 = np.linspace(bot, top, num=N)
s2 = np.linspace(bot, top, num=N)
s = []
for i in range(N):
    for j in range(N):
        s.append([s1[i], s2[j]])

num_cols = 2
num_rows = len(s)

# initialize random target output labels
# np.random.choice([0, 1], size=(10,), p=[1./3, 2./3])
T = num_rows
z = 100
t1 = [0]*z + [1]*(T-z)
t2 = [0]*(T-z) + [1]*z
np.random.shuffle(t1)
np.random.shuffle(t2)
tgt = []
for i in range(num_rows):
    tgt.append([t1[i], t2[i]])

# normalization of sigmoid signal
norm_out = []
for r in range(num_rows):
    sums = 0.0
    for c in range(num_cols):
        sums += math.exp(s[r][c])
    smp = []
    for c in range(num_cols):
        smp.append(math.exp(s[r][c]) / sums)
    norm_out.append(smp)

# d_k and l_k function calculation
dk = []
lk = []
npos = 0
smooth_fp = 0.
smooth_tp = 0.
alpha_param = 50
ebeta = math.exp(-0.1)
for r in range(num_rows):
    nowd = []
    nowl = []
    for c in range(num_cols):
        tmp = (num_cols - 1) * norm_out[r][c] / (1.0 - norm_out[r][c])
        # dk calc
        nowd.append(math.log(1.0 / tmp))
        # lk calc
        nowl.append(1.0 / (1.0 + math.pow(tmp, alpha_param) * ebeta))

        if tgt[r][c] > 0:
        # count all positive labels: np = |C_k| = TP_k + FN_k
            npos = npos + 1
            smooth_tp = smooth_tp + 1.0 - nowl[c]
        else:
            smooth_fp = smooth_fp + 1.0 - nowl[c]



    dk.append(nowd)
    lk.append(nowl)

# smooth microF1



# target distribution plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(column(s, 0), column(s, 1), column(s, 1), c='r', marker='o')
# #ax.scatter(column(s, 0), column(s, 1), column(s, 2), c='b', marker='o')
# #ax.scatter(column(s, 0), column(s, 1), column(dk, 1), c='b', marker='o')
#
# ax.set_xlabel('$\sigma_1$ axis')
# ax.set_ylabel('$\sigma_2$ axis')
# ax.set_zlabel('$target_k$ axis')
#
# plt.rc('text', usetex=True)
# plt.show()


# dk distribution plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(column(s, 0), column(s, 1), column(dk, 1), c='r', marker='o')
ax.scatter(column(s, 0), column(s, 1), column(lk, 1), c='b', marker='o')

ax.set_xlabel('$\sigma_1$ axis')
ax.set_ylabel('$\sigma_2$ axis')
ax.set_zlabel('$d_k$ axis')

plt.rc('text', usetex=True)
plt.show()


# micro F1 test
Num = 100
loss = np.linspace(bot, top, num=Num)
mF = []
for i in range(Num):
    f = 100.0 - 200.0 * 8 * 10 * (1.0-loss[i])/(8*(10 + 80 * (1.0 - loss[i])))
    mF.append(f)

plt.plot(loss, mF)
plt.show()


## microF1 function calculation
# for i in xrange(N):
# for
# j in xrange(N):
#
#
# fmicro = [2, 3, 1, 4, 3, 2, 5, 3]
#
#
