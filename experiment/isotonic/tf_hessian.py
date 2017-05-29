"""
Ref: http://runopti.github.io/blog/2016/07/07/HessianComp/
"""

import tensorflow as tf
import matplotlib as plt
import numpy as np
import math


def getHessian(dim):
    # Each time getHessian is called,
    # we create a new graph so that the default graph (which exists a priori)
    # won't be filled with old ops.
    g = tf.Graph()
    with g.as_default():
        # First create placeholders for inputs: A, b, and c.
        A = tf.placeholder(tf.float32, shape=[dim, dim])
        b = tf.placeholder(tf.float32, shape=[dim, 1])
        c = tf.placeholder(tf.float32, shape=[1])

        # Define our variable
        x = tf.Variable(np.float32(np.repeat(1, dim).reshape(dim, 1)))
        # Construct the computational graph for quadratic function: f(x) = 1/2 * x^t A x + b^t x + c
        fx = 0.5 * tf.matmul(tf.matmul(tf.transpose(x), A), x) + tf.matmul(tf.transpose(b), x) + c

        # Get gradients of fx with repect to x
        dfx = tf.gradients(fx, x)[0]
        # Compute hessian
        for i in range(dim):
            # Take the i th value of the gradient vector dfx
            # tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
            dfx_i = tf.slice(dfx, begin=[i, 0], size=[1, 1])
            # Feed it to tf.gradients to compute the second derivative.
            # Since x is a vector and dfx_i is a scalar,
            # this will return a vector : [d(dfx_i) / dx_i , ... , d(dfx_n) / dx_n]
            # whenever we use tf.gradients,
            # make sure you get the actual tensors by putting [0] at the end
            ddfx_i = tf.gradients(dfx_i, x)[0]
            if i == 0:
                hess = ddfx_i
            else:
                hess = tf.concat(1, [hess, ddfx_i])
            # Instead of doing this, you can just append each element to a list,
            # and then do tf.pack(list_object) to get the hessian matrix too.
            # I'll use this alternative in the second example.

        # Before we execute the graph, we need to initialize all the variables we defined
        init_op = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init_op)
            # We need to feed actual values into the computational graph that we created above.
            feed_dict = {A: np.float32(np.repeat(2, dim * dim).reshape(dim, dim)),
                         b: np.float32(np.repeat(3, dim).reshape(dim, 1)), c: [1]}
            # sess.run() executes the graph. Here, "hess" will be calculated with the values in "feed_dict".
            print(sess.run(hess, feed_dict))


if __name__ == "__main__":
    getHessian(5)