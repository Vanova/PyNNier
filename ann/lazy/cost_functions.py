import numpy as np
import nonlinearity as nonl


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
        return (a - y) * nonl.sigmoid_prime(z)


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
        norms = nonl.softmax(a)
        # calculate class loss function l
        d = np.log(1. / (nclass - 1) * (1. / norms - 1.) + 1e-6)
        l = 1.0 / (1.0 + np.exp(-MFoMCost.alpha * d - MFoMCost.beta))
        # smooth approximation
        yneg = np.logical_not(y)
        smooth_fp = np.sum((1.0 - l) * yneg)
        smooth_tp = np.sum((1.0 - l) * y)

        # Jacobian
        delta_l = MFoMCost.alpha * l * (1.0 - l)
        sum_jac = np.zeros((nsamples, nclass))
        # TODO check the sum inference of Jacobian
        # weighted jacobian on every sample
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
        norms = nonl.softmax(a)
        # calculate class loss function l
        d = np.log(1. / (nclass - 1) * (1. / norms - 1.) + 1e-6)
        # TODO L FUNCTION INVERSES SCORES!!!
        l = 1.0 / (1.0 + np.exp(-MFoMCost.alpha * d - MFoMCost.beta))
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


class UnitsvsZerosMFoMCost(object):
    alpha = 1.0
    beta = 0.0

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        :param a: network output on batch or the whole dataset, 2D array, smp x dim
        :param y: binary multi-labels, 2D array, smp x dim
        """
        l = UnitsvsZerosMFoMCost.class_loss_scores(a)
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
        nsamples = y.shape[0]
        nclass = y.shape[1]
        npos = np.sum(y == 1)
        # softmax normalization: ssigma in report
        norms = nonl.softmax(a)

        # choose the scores corresponding to unit and zero labels
        # units-vs-zeros misclassification measure
        d = np.zeros((nsamples, nclass))
        for r in xrange(nsamples):
            for c in xrange(nclass):
                d[r, c] = -a[r, c]
                anti = 1.
                if y[r, c] > 0:
                    # antimodels are zeros
                    nanti = np.sum(y[r] == 0)
                    if nanti:
                        anti = 1. / nanti * np.sum(np.exp(a[r][y[r] == 0]))
                else:
                    # antimodels are units
                    nanti = np.sum(y[r] == 1)
                    if nanti:
                        anti = 1. / nanti * np.sum(np.exp(a[r][y[r] == 1]))
                d[r, c] += np.log(anti)

        # calculate class loss function l
        # d = np.log(1. / (nclass - 1) * (1. / norms - 1.) + 1e-6)
        l = 1.0 / (1.0 + np.exp(-UnitsvsZerosMFoMCost.alpha * d - UnitsvsZerosMFoMCost.beta))

        # smooth approximation
        yneg = np.logical_not(y)
        smooth_fp = np.sum((1.0 - l) * yneg)
        smooth_tp = np.sum((1.0 - l) * y)

        # Jacobian
        delta_l = UnitsvsZerosMFoMCost.alpha * l * (1.0 - l)
        sum_jac = np.zeros((nsamples, nclass))
        # TODO check the sum inference of Jacobian
        # weighted jacobian on every sample
        count = 0
        for dl_smp, snorm_smp, y_smp, a_smp in zip(delta_l, norms, y, a):
            sum_jac[count] = UnitsvsZerosMFoMCost._weighted_jacobian(dl_smp, snorm_smp, y_smp, a_smp,
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
        # TODO fix to Units-vs-zeros!!!
        nclass = a.shape[1]
        # softmax normalization: ssigma in report
        norms = nonl.softmax(a)
        # calculate class loss function l
        d = np.log(1. / (nclass - 1) * (1. / norms - 1.) + 1e-6)
        # d = -a + 0.5
        l = 1.0 / (1.0 + np.exp(-MFoMCost.alpha * d - MFoMCost.beta))
        # l = 1. - a  # i.e. sigmoids
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


def step(a, threshold=0.0):
    """Heaviside step function:
    a < threshold = 0, else 1.
    :param a: array
    :return: array
    """
    res = np.zeros_like(a)
    res[a < threshold] = 0
    res[a >= threshold] = 1
    return res
