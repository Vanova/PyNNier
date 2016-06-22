import nltk
import lib
import itertools
import timeit
import numpy as np
import model.rnn_numpy as rp
import optimizer

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[lib.SENTENCE_START_TOKEN]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[lib.SENTENCE_END_TOKEN]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[lib.UNKNOWN_TOKEN]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[lib.UNKNOWN_TOKEN]:
            # Draw samples from a multinomial distribution, with probabilit y_0
            samples = np.random.multinomial(1, next_word_probs[0][-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str


# Tokenize the sentences into words
tokenized_sentences, word_to_index, index_to_word, vocab = lib.preprocess_data('data/war_n_peace.txt')

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

# Print training data example
x_example, y_example = X_train[17], y_train[17]
print "x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example)
print "\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example)

### Test prediction
np.random.seed(777)
model = rp.RNNNumpy(lib.VOCABULARY_SIZE)
o, s = model.forward_propagation(X_train[10])
print o.shape
print o

predictions = model.predict(X_train[10])
print predictions.shape
print predictions

### Test loos function
# Limit to 1000 examples to save time
print "Expected Loss for random predictions: %f" % np.log(lib.VOCABULARY_SIZE)
print "Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000])

### Test running time for one step
np.random.seed(10)
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

num_sentences = 15
senten_min_length = 7

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent)

# TODO infere gradience
# e.g. fortification cup it pleasure rubles convince hospital
# opposition : burn operating obeyed lord
