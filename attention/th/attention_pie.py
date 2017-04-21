"""
attention_pie.py
~~~~~~~~~~~~~~~~
This is a 'pie' from CNN + RNN(LST or GRU) + Attention models.
Test on MNIST dataset
Model will be mostly adapted for speech data.

Ref:
    https://github.com/Vanova/CS-224U-Project
    https://github.com/Vanova/seq2seq
    https://github.com/Vanova/recurrentshop
"""
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils

np.random.seed(777)
assert K.backend() == 'theano'
assert K.image_dim_ordering() == 'th'
K.set_image_dim_ordering('th')

# train setting
batch_size = 128
nb_epoch = 12
nb_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28
input_length = img_rows
input_dim = img_cols
# network output
output_length = 1
output_dim = nb_classes
samples = batch_size

def prepare_mnist():
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_train = Y_train.reshape(Y_train.shape[0], output_length, nb_classes)
    Y_test = Y_test.reshape(Y_test.shape[0], output_length, nb_classes)
    return X_train, Y_train, X_test, Y_test

def test_AttentionSeq2Seq():
    x = np.random.random((samples, input_length, input_dim))
    y = np.random.random((samples, output_length, output_dim))

    models = []
    models += [
        AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim))]
    models += [
        AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim),
                         depth=2)]
    models += [
        AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim),
                         depth=3)]

    models[0].compile(loss='mse', optimizer='sgd')
    m = models[0].fit(x, y, nb_epoch=1)
    # m = models[0].train_on_batch(x, y)
    print(m)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = prepare_mnist()
    model = AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim),
                             bidirectional=False)
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adadelta',
    #               metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    pred = model.predict(X_test)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
