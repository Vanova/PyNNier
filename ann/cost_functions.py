import numpy as np


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        Input:
        Output:
        """

        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a - y)


class MFoMCost(object):
    alpha = 1.0
    beta = 0.0

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        :param a: network output on batch or the whole dataset, 2D array, smp x dim
        :param y: binary multi-labels, 2D array, smp x dim
        """
        l = MFoMCost.class_loss_scores(a)
        # smooth approximation
        yneg = np.logical_not(y)
        npos = (y == 1).sum()
        smooth_fp = np.sum((1.0 - l) * yneg)
        smooth_tp = np.sum((1.0 - l) * y)
        f1_error = 100.0 - 200.0 * smooth_tp / \
                           (smooth_tp + smooth_fp + npos)
        return f1_error

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer,
        i.e. partial derivatives \partial F_x /\partial z.
        F = 2TP/(FP + TP + npos),
        npos = TP + FN - number of positive samples
        :param z: network affine transformations
        :param a: network output on batch, 2D array, smp x dim
        :param y: binary multi-labels, 2D array, smp x dim
        :return diff: list of gradients on each sample prediction
        """
        nclass = y.shape[1]
        nsamples = y.shape[0]
        npos = (y == 1).sum()
        # softmax normalization: ssigma in report
        norms = softmax(a)
        # calculate class loss function l
        d = (nclass - 1) * norms / (1.0 - norms)
        ebeta = np.exp(-MFoMCost.beta)
        l = 1.0 / (1.0 + np.power(d, MFoMCost.alpha) * ebeta)

        # smooth approximation
        yneg = np.logical_not(y)
        smooth_fp = np.sum((1.0 - l) * yneg)
        smooth_tp = np.sum((1.0 - l) * y)

        # Jacobian
        delta_l = MFoMCost.alpha * l * (1.0 - l)
        sum_jac = np.zeros((nsamples, nclass))
        # TODO check the sum inference of Jacobian
        # weighted jacobian on every sample
        # for dl_smp, snorm_smp, y_smp, a_smp in zip(delta_l, norms, y, a):
        #     sum_jac += MFoMCost.weighted_jacobian(dl_smp, snorm_smp, y_smp, a_smp,
        #                                       npos + smooth_fp, smooth_tp)
        count = 0
        for dl_smp, snorm_smp, y_smp, a_smp in zip(delta_l, norms, y, a):
            sum_jac[count] = MFoMCost._weighted_jacobian(dl_smp, snorm_smp, y_smp, a_smp,
                                                         npos + smooth_fp, smooth_tp)
            count += 1
        return 2. * sum_jac / (smooth_fp + smooth_tp + npos) ** 2

    @staticmethod
    def class_loss_scores(a):
        """
        Calculate class loss function with "ONE-vs-OTHERS"
        misclassification measure
        :param a: network output signal, e.g. sigmoid scores
        """
        nclass = a.shape[1]
        # softmax normalization: ssigma in report
        norms = softmax(a)
        # calculate class loss function l
        # TODO check the loss inference!!! L FUNCTION INVERSE SCORES!!!
        d = (nclass - 1) * norms / (1.0 - norms)
        ebeta = np.exp(-MFoMCost.beta)
        l = 1.0 / (1.0 + np.power(d, MFoMCost.alpha) * ebeta)
        return l


    @staticmethod
    def _weighted_jacobian(dl_smp, snorm_smp, y_smp, a_smp, scale_pos, scale_neg):
        nclass = snorm_smp.shape[0]
        diag_elements = dl_smp / (1.0 - snorm_smp)
        diag = np.diag(diag_elements)
        off_diag = np.tile(diag_elements, (nclass, 1))
        off_diag = off_diag * snorm_smp.reshape((-1, 1))
        # sigmoid derivative
        a_prime = a_smp * (1.0 - a_smp)
        # TODO explore diag-offdiag MFoM loss
        # J = -diag * a_prime.reshape((-1, 1))
        J = (-diag + off_diag) * a_prime.reshape((-1, 1))
        # weighting the Jacobian
        y_pos = (y_smp == 1.0)
        y_neg = np.invert(y_pos)
        W = scale_pos * y_pos - scale_neg * y_neg
        return np.dot(J, W.reshape((-1, 1))).T




### Other functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_linear(z, a=1., b=0.):
    return 1.0 / (1.0 + np.exp(-a * z + b))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    """
    Normalization across the row vector
    :param z: 2D array, smp x dim
    """
    zt = z.transpose()
    x = zt - np.max(zt, axis=0)  # safe explosion trick
    p = np.exp(x) / np.sum(np.exp(x), axis=0)
    return p.transpose()


def step(a, threshold=0.0):
    """Heaviside step function:
    a < threshold = 0, else 1.
    :param a: array
    :return: array
    """
    # TODO refactor!!!
    res = np.zeros_like(a)
    res[a < threshold] = 0
    res[a >= threshold] = 1
    return res
