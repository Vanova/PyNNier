import timeit

import numpy as np

import lib
import model.rnn_numpy as rp
from rnn.nlp.model import optimizer


def generate_sentence(model, num_sentences=1, senten_min_length=3):
    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = generating(model)
        print " ".join(sent)
    return sent


def generating(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[lib.SENTENCE_START_TOKEN]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[lib.SENTENCE_END_TOKEN]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[lib.UNKNOWN_TOKEN]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[lib.UNKNOWN_TOKEN]:
            # Draw samples from a multinomial distribution, with probabilit y_t
            samples = np.random.multinomial(1, next_word_probs[0][-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str


# Tokenize the sentences into words
tokenized_sentences, word_to_index, index_to_word, vocab = lib.preprocess_data('data/war_n_peace.txt')

# Create the training data: map words into indeces
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

# Print training data example
x_example, y_example = X_train[17], y_train[17]
print "x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example)
print "\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example)

### Test running time for one step
np.random.seed(777)
model = rp.RNNNumpy(lib.VOCABULARY_SIZE)

t1 = timeit.default_timer()
model.numpy_sdg_step(X_train[10], y_train[10], 0.005)
t2 = timeit.default_timer()
print "Time of one sgd step %f" % ((t2 - t1) * 1000)

# Train on a small subset of the data to see what happens
np.random.seed(10)
model = rp.RNNNumpy(lib.VOCABULARY_SIZE)
losses = optimizer.train_with_sgd_numpy(model, X_train[:100], y_train[:100], nepoch=1, evaluate_loss_after=1)
print(losses)

# produce sentences from model
sent = generate_sentence(model, num_sentences=3, senten_min_length=7)
