import numpy as np
import tensorflow as tf

tf.reset_default_graph()

sess = tf.InteractiveSession()

PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = 2 * encoder_hidden_units

# placeholders
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_input_len')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

# embeddings matrix
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1., 1.), dtype=tf.float32)
# feed raw inputs through the embedding matrix
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

# define encoder
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple

encoder_cell = LSTMCell(encoder_hidden_units)

((encoder_fw_outputs, encoder_bw_outputs),
 (encoder_fw_final_state, encoder_bw_final_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                                                      cell_bw=encoder_cell,
                                                                                      inputs=encoder_inputs_embedded,
                                                                                      sequence_length=encoder_inputs_length,
                                                                                      dtype=tf.float32,
                                                                                      time_major=True))

# bidirectional step
encoder_final_state_h = tf.concat(1, (encoder_fw_final_state.h, encoder_bw_final_state.h))
encoder_final_state_c = tf.concat(1, (encoder_fw_final_state.c, encoder_bw_final_state.c))
# tuple of cell states and hidden states are used by tuple LSTM
encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)

# decoder
decoder_cell = LSTMCell(decoder_hidden_units)
encoder_max_time, batch_sz = tf.unstack(tf.shape(encoder_inputs))
decoder_length = encoder_inputs_length + 3

# output projection
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1., 1.), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

# create padded inputs for the decoder from the word embeddings
eos_time_slice = tf.ones([batch_sz], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_sz], dtype=tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)


# define attention

def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_length)  # all False at the initial step
    # end of sentence
    initial_input = eos_step_embedded
    # last time steps cell state
    initial_cell_state = encoder_final_state
    # none
    initial_cell_output = None
    initial_loop_state = None  # don't need to pass any additional information
    return (initial_elements_finished, initial_input,
            initial_cell_state, initial_cell_output, initial_loop_state)


# attention mechanism --choose which previously generated token to pass as input in the next time step
def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        # dot product between previous ouput and weights, then + biases
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        # Logits simply means that the function operates on the unscaled output of
        # earlier layers and that the relative scale to understand the units is linear.
        # It means, in particular, the sum of the inputs may not equal 1, that the values are not probabilities
        # (you might have an input of 5).
        # prediction value at current time step

        # Returns the index with the largest value across axes of a tensor.
        # Attention focusing
        prediction = tf.argmax(output_logits, axis=1)
        # embed prediction for the next input
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input

    elements_finished = (time >= decoder_length)  # this operation produces boolean tensor of [batch_size]
    # defining if corresponding sequence has ended
    # Computes the "logical and" of elements across dimensions of a tensor.
    finished = tf.reduce_all(elements_finished)  # -> boolean scalar
    # Return either fn1() or fn2() based on the boolean predicate pred.
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)

    # set previous to current
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished, input, state,
            output, loop_state)


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:  # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)


# Creates an RNN specified by RNNCell cell and loop function loop_fn.
# This function is a more primitive version of dynamic_rnn that provides more direct access to the
# inputs each iteration. It also provides more control over when to start and finish reading the sequence,
# and what to emit for the output.
# ta = tensor array
decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.pack()

# to convert output to human readable prediction
# we will reshape output tensor
# Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
# reduces dimensionality
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
# flattened output tensor
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
# pass flattened tensor through decoder
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
# prediction vals
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

# final prediction
decoder_prediction = tf.argmax(decoder_logits, 2)
