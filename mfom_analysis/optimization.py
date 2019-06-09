"""
Analysis of optimization approaches for MFoM-microF1 to see
how smooth micro-F1 optimize discrete F1 regarding different threshold 't'

We keep scores fixed and vary only 'a' and 'b' parameters in smooth MFoM-microF1
"""
import mfom.utils.dcase_scores as dcase
import metrics.metrics as metr
import tensorflow as tf

_EPSILON = 1e-7

# P_df = dcase.read_dcase('data/test_scores/results_fold2.txt')
# Y_df = dcase.read_dcase('data/test_scores/y_true_fold2.txt')
P_df = dcase.read_dcase('data/all/train_scores_fold1.txt')
Y_df = dcase.read_dcase('data/all/train_Y_fold1.txt')
P_df_test = dcase.read_dcase('data/all/test_scores_fold1.txt')
Y_df_test = dcase.read_dcase('data/all/test_Y_fold1.txt')

###
# calculate discrete F1
###
bin_P = metr.step(P_df.values, threshold=0.5)
mf = metr.micro_f1(y_true=Y_df.values, y_pred=bin_P)
print('Discrete Train micro-F1: %.4f' % mf)

bin_P = metr.step(P_df_test.values, threshold=0.5)
mf = metr.micro_f1(y_true=Y_df_test.values, y_pred=bin_P)
print('Discrete Test micro-F1: %.4f' % mf)


###
# approximate with batch smooth MFoM-F1
###
def _uvz_loss_scores(y_true, y_pred, alpha, beta, eta, is_training=True):
    if is_training:
        y_pred = tf.clip_by_value(y_pred, _EPSILON, 1.0 - _EPSILON)
        y_pred = eta * y_pred
        y_neg = 1 - y_true

        # Kolmogorov log average of unit labeled models
        unit_avg = y_true * tf.exp(y_pred)  # [smp, dim]
        # average over non-zero elements
        unit_avg = tf.log(_non_zero_mean(unit_avg))  # [smp, 1]
        # Kolmogorov log average of zero labeled models
        zeros_avg = y_neg * tf.exp(y_pred)
        # average over non-zero elements
        zeros_avg = tf.log(_non_zero_mean(zeros_avg))
        # misclassification measure, optimized
        d = -y_pred + 1. / eta * (y_neg * unit_avg + y_true * zeros_avg)
    else:
        d = -y_pred + 0.5 #1. / eta * 0.5

    # calculate class loss function l
    l = tf.sigmoid(alpha * d + beta)
    return l


def _non_zero_mean(x):
    mask = tf.greater(tf.abs(x), 0)
    n = tf.reduce_sum(tf.cast(mask, 'float32'), axis=1, keep_dims=True)
    return tf.reduce_sum(x, axis=-1, keep_dims=True) / n


def next_batch(batch_size):
    # while 1:
    # np.random.shuffle()
    num_batch = num_smp // batch_size
    for i in xrange(num_batch):
        batch_x = P_df.values[i * batch_size: (i + 1) * batch_size]
        batch_y = Y_df.values[i * batch_size: (i + 1) * batch_size]
        yield batch_x, batch_y


# Parameters
learning_rate = 0.01
training_epochs = 300
batch_size = 10  # 30
display_step = 1

# init variables
num_smp, dim = P_df.shape
G = tf.placeholder('float', [None, dim])
Y = tf.placeholder('float', [None, dim])
a = tf.Variable(tf.random_normal([dim]), trainable=True)
b = tf.Variable(tf.random_normal([dim]), trainable=True)
# TODO add this for training!!!
eta_var = tf.Variable(tf.random_normal([dim]), trainable=True)


# MFoM scores
# TODO change for training/testing!!!
# L = _uvz_loss_scores(y_true=Y, y_pred=G, alpha=a, beta=b, is_training=False)
# P = 1. - L

# smooth F1
# numen = 2. * tf.reduce_sum(P * Y)
# denum = tf.reduce_sum(P + Y)
# smoothF = 1. - tf.divide(numen, denum)


# Smooth F1 for training
def smooth_F1(g, y, alpha=None, beta=None, eta=None, is_training=True):
    """
    l_scores: MFoM loss scores, tf tensor
    y: target labels, tf tensor
    :return: smoothed F1 value
    """
    if not alpha and not beta and not eta and is_training:
        # train 'alpha', 'beta' and 'eta'
        l_scores = _uvz_loss_scores(y_true=y, y_pred=g, alpha=a, beta=b, eta=eta_var, is_training=True)
    else:
        # apply trained 'alpha' and 'beta'
        l_scores = _uvz_loss_scores(y_true=y, y_pred=g,
                                    alpha=alpha, beta=beta, eta=eta,
                                    is_training=False)

    p = 1. - l_scores
    numen = 2. * tf.reduce_sum(p * y)
    denum = tf.reduce_sum(p + y)
    smoothF = tf.divide(numen, denum)
    return smoothF


# Network optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# TODO note: False because we don't apply UvZ strategy, train only a, b, eta
trainF1 = smooth_F1(g=G, y=Y, alpha=a, beta=b, eta=eta_var, is_training=False)  # TODO try False
train_op = optimizer.minimize(1. - trainF1)
# Initializing the variables
init = tf.global_variables_initializer()

# run optimization
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = num_smp // batch_size
        # Loop over all batches
        for batch_x, batch_y in next_batch(batch_size):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, 1. - trainF1], feed_dict={G: batch_x,
                                                                 Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch smooth F1 (avg batch):", '%04d' % (epoch + 1), "cost={:.9f}".format((1. - avg_cost) * 100.))

        if epoch % (display_step + 4) == 0:
            testF1 = smooth_F1(g=G, y=Y, alpha=a, beta=b, eta=eta_var, is_training=False)
            print("smooth Train F1 (whole dataset):", (testF1.eval({G: P_df, Y: Y_df})) * 100.)
            print("smooth Test F1 (whole dataset):", (testF1.eval({G: P_df_test, Y: Y_df_test})) * 100.)
            print

    print("Optimization Finished!")

    # Testing
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    testF1 = smooth_F1(g=G, y=Y, alpha=a, beta=b, is_training=False)
    print("smooth F1 (whole dataset):", (testF1.eval({G: P_df.values, Y: Y_df})) * 100.)
