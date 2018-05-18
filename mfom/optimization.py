"""
Analysis of optimization approaches for MFoM-microF1 to see
how smooth micro-F1 optimize discrete F1 regarding different threshold 't'

We keep scores fixed and vary only 'a' and 'b' parameters in smooth MFoM-microF1
"""
import mfom.utils.dcase_scores as dcase
import metrics.metrics as metr
import tensorflow as tf

_EPSILON = 1e-7

P_df = dcase.read_dcase('data/test_scores/results_fold2.txt')
Y_df = dcase.read_dcase('data/test_scores/y_true_fold2.txt')
###
# calculate discrete F1
###
bin_P = metr.step(P_df.values, threshold=0.5)
mf = metr.micro_f1(y_true=Y_df.values, y_pred=bin_P)
print('Discrete micro-F1: %.4f' % mf)


###
# approximate with batch smooth MFoM-F1
###
def _uvz_loss_scores(y_true, y_pred, alpha, beta, is_training=True):

    if is_training:
        y_pred = tf.clip_by_value(y_pred, _EPSILON, 1.0 - _EPSILON)
        y_neg = 1 - y_true
        # Kolmogorov log average of unit labeled models
        unit_avg = y_true * tf.exp(y_pred)              # [smp, dim]
        # average over non-zero elements
        unit_avg = tf.log(_non_zero_mean(unit_avg))     # [smp, 1]
        # Kolmogorov log average of zero labeled models
        zeros_avg = y_neg * tf.exp(y_pred)
        # average over non-zero elements
        zeros_avg = tf.log(_non_zero_mean(zeros_avg))
        # misclassification measure, optimized
        d = -y_pred + y_neg * unit_avg + y_true * zeros_avg
    else:
        d = -y_pred + 0.5

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
        batch_x = P_df.values[i * batch_size : (i+1) * batch_size]
        batch_y = Y_df.values[i * batch_size : (i+1) * batch_size]
        yield batch_x, batch_y


# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 10
display_step = 1

# init variables
num_smp, dim = P_df.shape
G = tf.placeholder('float', [None, dim])
Y = tf.placeholder('float', [None, dim])
a = tf.Variable(tf.random_normal([dim]), trainable=True)
b = tf.Variable(tf.random_normal([dim]), trainable=True)

# MFoM scores
# TODO change for training/testing!!!
L = _uvz_loss_scores(y_true=Y, y_pred=G, alpha=a, beta=b, is_training=False)
P = 1. - L

# smooth F1
numen = 2. * tf.reduce_sum(P * Y)
denum = tf.reduce_sum(P + Y)
smoothF = 1. - tf.divide(numen, denum)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(smoothF)
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
            _, c = sess.run([train_op, smoothF], feed_dict={G: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")


    # Test model
    # pred = smF  # Apply softmax to logits
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("smooth F1 (whole dataset):", (1. - smoothF.eval({G: P_df.values, Y: Y_df})) * 100.)