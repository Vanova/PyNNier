import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout, Lambda, noise, GlobalAveragePooling2D, BatchNormalization, Merge, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot
import cv2
import pylab as pl
import matplotlib.cm as cm

assert K.backend() == 'theano'
assert K.image_dim_ordering() == 'th'
K.set_image_dim_ordering('th')

# assert K.backend() == 'tensorflow'
# assert K.image_dim_ordering() == 'tf'
# K.set_image_dim_ordering('tf')


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """

    def f(X):
        b, ch, r, c = X.shape  # batch, channel, row, column
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1)), (0, half))
        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:, i:i + ch, :, :]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def global_average_pooling(x):
    # (samples, channels, rows, cols)
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def attention_control(args):
    # Why do we need dense_2 here!?
    x, dense_2 = args
    find_att = K.reshape(x, (15, 15, 10))
    find_att = K.transpose(find_att[:, :, :]) # 10 x 15 x 15
    find_att = K.mean(find_att, axis=0) # 15 x 15
    # WTF ???
    find_att = find_att / K.sum(find_att, axis=0) # 15 x 15
    # TODO ??? maybe BUG: copy across channels, but he lose channel axis
    find_att = K.repeat_elements(find_att, 32, axis=0)
    find_att = K.reshape(find_att, (1, 32, 15, 15))

    # x, dense_2 = args
    # find_att = K.reshape(x, (15, 15, 10))
    # find_att = K.transpose(find_att[:, :, :])  # 10 x 15 x 15
    # # find average attention here across all feature maps
    # mean_att = K.mean(find_att, axis=0)  # 15 x 15
    # mean_att = K.reshape(mean_att, (1, 15, 15))
    # # copy attention mask across all feature maps
    # rep_mean_att = K.repeat_elements(mean_att, 32, axis=0)
    # focus = K.reshape(rep_mean_att, (1, 32, 15, 15))
    # return focus
    return find_att

def no_attention_control(args):
    x, dense_2 = args
    find_att = K.ones(shape=(1, 32, 15, 15))
    return find_att


def change_shape1(x):
    # (samples, channels, rows, cols) -> (cols, rows, channels, samples) -> (cols * rows, channels [, samples])
    # TODO: we lose sample axis here, if one sample in batch
    # x = K.reshape(K.transpose(x), (15 * 15, 32, batch_sz))
    x = K.reshape(K.transpose(x), (15 * 15, 32))
    return x


def att_shape(input_shape):
    return (input_shape[0][0], 32, 15, 15)


def att_shape2(input_shape):
    return input_shape[0][0:4]


# def mnist_gap(is_gap=False):
#     nb_filters = 32
#     kernel_size = (3, 3)
#     pool_size = (3, 3)
#     input_shape = (28, 28, 1)
#     nb_classes = 10
#     # 1st conv
#     model = Sequential()
#     model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
#                             border_mode='valid', input_shape=input_shape, name='conv_1'))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2), name='convmax_1'))
#     # model.add(crosschannelnormalization(name="convpool_1"))
#     model.add(BatchNormalization(axis=3, mode=0, name="convpool_1"))
#     # 2nd conv
#     model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], name='conv_2'))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2), name='convmax_2'))
#     # model.add(crosschannelnormalization(name="convpool_2"))
#     model.add(BatchNormalization(axis=3, mode=0, name="convpool_2"))
#     model.add(Dropout(0.25))
#     # 1st dense layer
#     model.add(Flatten())
#     model.add(Dense(128, name='dense_1'))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     # 2nd dense layer
#     model.add(Dense(nb_classes, name='dense_2'))
#     model.add(Activation('softmax'))
#
#     return model


def mnist_attention(inc_noise=False, attention=True, grad_cam_train=True):
    # The map equation: (width - kernel_size + 2*pad)/stride +1
    # make layers
    inputs = Input(shape=(1, image_size, image_size), name='input') # 1 x 30 x 30

    conv_1a = Convolution2D(32, 3, 3, activation='relu', name='conv_1') # 32 x 28 x 28: [30 - 3 + 1]
    maxp_1a = MaxPooling2D((3, 3), strides=(2, 2), name='convmax_1') # 32 x 13 x 13: [(27 - 3) / 2 + 1]
    norm_1a = crosschannelnormalization(name="convpool_1") #BatchNormalization(axis=1, mode=0, name="convpool_1") # 32 x 13 x 13
    zero_1a = ZeroPadding2D((2, 2), name='convzero_1') # 32 x 17 x 17: [13 + 4]

    conv_2a = Convolution2D(32, 3, 3, activation='relu', name='conv_2') # 32 x 15 x 15: [17 - 3 + 1]
    maxp_2a = MaxPooling2D((3, 3), strides=(2, 2), name='convmax_2') # 32 x 7 x 7: [(15 - 3) / 2 + 1]
    norm_2a = crosschannelnormalization(name="convpool_2") #BatchNormalization(axis=1, mode=0, name="convpool_2") # 32 x 7 x 7
    zero_2a = ZeroPadding2D((2, 2), name='convzero_2') # 32 x 11 x 11

    # TODO try Keras GlobalAveragePooling2D: (samples, rows, cols, channels) -> (nb_samples, channels), for 'tf'
    # (samples, channels, rows, cols) -> (nb_samples, channels), for 'th'
    dense_1a = Lambda(global_average_pooling, output_shape=global_average_pooling_shape, name='dense_1')
    # dense_1a = GlobalAveragePooling2D(name='GAP')
    dense_2a = Dense(10, activation='softmax', init='uniform', name='dense_2')

    # make actual model
    if inc_noise:
        inputs_noise = noise.GaussianNoise(2.5)(inputs)
        input_pad = ZeroPadding2D((1, 1), input_shape=(1, image_size, image_size), name='input_pad')(inputs_noise)
    else:
        input_pad = ZeroPadding2D((1, 1), input_shape=(1, image_size, image_size), name='input_pad')(inputs)

    conv_1 = conv_1a(input_pad)
    conv_1 = maxp_1a(conv_1)
    conv_1 = norm_1a(conv_1)
    conv_1 = zero_1a(conv_1)

    conv_2_x = conv_2a(conv_1)
    conv_2 = maxp_2a(conv_2_x)
    conv_2 = norm_2a(conv_2)
    conv_2 = zero_2a(conv_2)
    conv_2 = Dropout(0.5)(conv_2)

    dense_1 = dense_1a(conv_2) # GAP on CNN feature maps
    dense_2 = dense_2a(dense_1) # dense layer + softmax

    if grad_cam_train:
        conv_shape1 = Lambda(change_shape1, output_shape=(32,), name='chg_shape')(conv_2_x)
        find_att = dense_2a(conv_shape1)

        if attention:
            find_att = Lambda(attention_control, output_shape=att_shape, name='att_con')([find_att, dense_2]) # find_att = [1 x 32 x 15 x 15]
        else:
            find_att = Lambda(no_attention_control, output_shape=att_shape, name='att_con')([find_att, dense_2])

        zero_3a = ZeroPadding2D((1, 1), name='convzero_3')(find_att) # 1 x 32 x 17 x 17
        apply_attention = Merge(mode='mul', name='attend')([zero_3a, conv_1]) # zero_3a = 1 x 32 x 17 x 17; conv_1 = 1 x 32 x 17 x 17

        conv_3 = conv_2a(apply_attention) # 32 x 15 x 15: [17 - 3 + 1]
        conv_3 = maxp_2a(conv_3) # 32 x 7 x 7: [(15 - 3) / 2 + 1]
        conv_3 = norm_2a(conv_3)
        conv_3 = zero_2a(conv_3) # 32 x 11 x 11

        dense_3 = dense_1a(conv_3)
        dense_4 = dense_2a(dense_3)

        # dense_3 = Flatten()(apply_attention)
        # dense_4 = dense_2a(dense_3)
        model = Model(input=inputs, output=dense_4)
    else:
        model = Model(input=inputs, output=dense_2)

    return model

# def visualize_cam(model, input_img):
#     """
#     Class activation map visualization
#     """
#     # model = load_model(model_path)
#     # image = cv2.imread(img_path, 1)
#     # fig = plt.figure(figsize=plt.figaspect(0.5))
#     # plt.subplot(1, 1, 1)
#     # plt.imshow(X_train[0][0])
#     width, height, _ = input_img.shape
#
#     # Reshape to the network input shape (3, w, h).
#     # img = np.array([np.transpose(np.float32(input_img), (2, 0, 1))])
#     img = np.array([input_img])
#
#     # Get the 512 input weights to the softmax.
#     class_weights = model.get_layer('conv_2').get_weights()[0] # model.layers[-1].get_weights()[0]
#     final_conv_layer = model.get_layer('conv_2')
#
#     get_output = K.function([model.layers[0].input, K.learning_phase()], [final_conv_layer.output, model.layers[-1].output])
#     [conv_outputs, predictions] = get_output([img, 0])
#     conv_outputs = conv_outputs[0, :, :, :]
#
#     # Create the class activation map.
#     cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])
#     for i, w in enumerate(class_weights[:, 1]):
#         cam += w * conv_outputs[i, :, :]
#     print "predictions", predictions
#     cam /= np.max(cam)
#     cam = cv2.resize(cam, (height, width))
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#     heatmap[np.where(cam < 0.2)] = 0
#     img = heatmap * 0.5 + input_img
#     cv2.imwrite("cam_img.png", img)
#
#     fig = plt.figure(figsize=plt.figaspect(0.5))
#     plt.subplot(1, 1, 2)
#     plt.imshow(input_img[0])
#     plt.subplot(1, 1, 2)
#     plt.imshow(img[0])


if __name__ == "__main__":
    ### TH
    # TODO fix
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    image_size = np.shape(X_train[0])[1]
    X_train.shape = (len(X_train), 1, image_size, image_size)
    X_test.shape = (len(X_test), 1, image_size, image_size)
    X_train /= 255
    X_test /= 255
    y_trainCAT = to_categorical(y_train)
    y_testCAT = to_categorical(y_test)

    # train and save model
    model = mnist_attention()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    model_history = model.fit(X_train, y_trainCAT, batch_size=1, validation_data=(X_test, y_testCAT), nb_epoch=3)
    score = model.evaluate(X_test, y_testCAT, verbose=0)
    model.save("mnist_%0.2f" % score[1])


    ### TF
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # image_size = np.shape(X_train[0])[1]
    # X_train.shape = (len(X_train), image_size, image_size, 1)
    # X_test.shape = (len(X_test), image_size, image_size, 1)
    # X_train /= 255
    # X_test /= 255
    # y_trainCAT = to_categorical(y_train)
    # y_testCAT = to_categorical(y_test)
    #
    # # train and save model
    # model = mnist_gap()
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.summary()
    # model_history = model.fit(X_train, y_trainCAT, batch_size=128, validation_data=(X_test, y_testCAT), nb_epoch=3)
    # score = model.evaluate(X_test, y_testCAT, verbose=0)
    # model.save("mnist_%0.2f" % score[1])

    # visualize Class Activation Map
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # plt.subplot(1, 1, 1)
    # plt.imshow(np.squeeze(X_train[0]))
    # pl.imshow(np.squeeze(X_train[0]), interpolation='nearest', cmap=cm.binary)
    # model = load_model("mnist_0.40")
    # plot(model, to_file='model.png', show_shapes=True)
    # visualize_cam(model, X_train[0])
