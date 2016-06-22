import itertools
import nltk
import csv
import numpy as np

VOCABULARY_SIZE = 8000
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"


def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print "Saved model parameters to %s." % outfile
   
def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])


def preprocess_data(text_filename):
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading text file %s ..." % text_filename
    with open(text_filename, 'rb') as f:  #
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        reader.next()
        # Split full text into sentences
        lst = []
        for x in reader:
            if x:
                low_case = nltk.sent_tokenize(x[0].decode('utf-8').lower())
                lst.append(low_case)
        sentences = itertools.chain(*lst)

        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))
    # split every sentence by words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(VOCABULARY_SIZE - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(UNKNOWN_TOKEN)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    # text info
    print "Using vocabulary size %d." % VOCABULARY_SIZE
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence (tokenized) after Pre-processing: '%s'" % tokenized_sentences[0]

    return tokenized_sentences, word_to_index, index_to_word, vocab