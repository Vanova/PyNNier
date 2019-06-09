from autograd import grad
from mfom.npmodel import objectives as obj

grad_mfom = grad(obj._uvz_loss_scores)

import numpy as np
X, Y = np.meshgrid(np.linspace(-10, 10, 10),
                   np.linspace(-10, 10, 10))

SX = obj.sigma(X)
SY = obj.sigma(Y)
# SY[SY >= 0.5] = 1
# SY[SY < 0.5] = 0

print(obj.mfom_eer_uvz(SY, SX))

grad_mfom(SY, SX)