"""
Approach is base on:
http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras
training/test = 50% / 50%
word embedding: from 500(int) -> 32(real)
vocabular = 5000
max movie review is set to 500: if more => truncated, if less => zero padding
"""

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model

# fix random seed for reproducibility
np.random.seed(777)

top_words = 5000
max_review_length = 500

def train():
    # load the dataset but only keep the top n words, zero the rest
    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(nb_words=top_words)
    # truncate and pad input sequences
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Dropout(0.2))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=2, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    model.save("imdb_%0.2f.pkl" % scores[1])

model = load_model("imdb_0.88.pkl")

indx = imdb.get_word_index()
test = "I feel bad It is so hot in the study room There is pool outside " + \
       "but I should work a lot of responsibility on me"
test = test.lower().split()
test_indx = [indx[w] for w in test]
print(test_indx)

pad_test = sequence.pad_sequences([test_indx], maxlen=max_review_length)
res = model.predict(pad_test)
print(res)


# def id2word(indx, id):
#     return indx.keys()[indx.values().index(id)]
#
# ws = [id2word(indx, id) for id in X_train[0]]
# print(Y_test[0])




