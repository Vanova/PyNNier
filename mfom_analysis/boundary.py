"""
Implementation of DNN with MFoM objective,
plot decision boundary on toy MULTI-LABEL dataset.
Check the difference from the regular objective functions
(e.g. binary cross-entropy).
Ref.:
https://plot.ly/scikit-learn/plot-multilabel
https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
http://www.magic-analytics.com/blog/visualize-decision-boundary-in-python
"""
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras.wrappers.scikit_learn import KerasClassifier
import keras.regularizers as regs
import keras.constraints as constraints
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import kmodel.mfom as kmfom
import kmodel.objectives as obj
import utils.metrics as MT
import visual.plotter as pltr

RANDOM_SEED = 777


def baseline_model(feat_dim, nclass):
    # input block
    feat_input = Input(shape=(feat_dim,), name='main_input')
    # layer 1
    x = Dense(10, name='dense1')(feat_input)
    x = Activation(activation='sigmoid', name='act1')(x)
    # layer 2
    x = Dense(10, name='dense2')(x)
    x = Activation(activation='sigmoid', name='act2')(x)
    # output layer
    x = Dense(nclass, name='pre_activation')(x)
    y_pred = Activation(activation='sigmoid', name='output')(x)

    b_model = Model(input=feat_input, output=y_pred)
    b_model.compile(loss='binary_crossentropy', optimizer='Adam')
    b_model.summary()
    return b_model


def mfom_model(feat_dim, nclass):
    # input block
    feat_input = Input(shape=(feat_dim,), name='main_input')
    # layer 1
    x = Dense(10, name='dense1')(feat_input)
    x = Activation(activation='sigmoid', name='act1')(x)
    # layer 2
    x = Dense(10, name='dense2')(x)
    x = Activation(activation='sigmoid', name='act2')(x)
    # layer 3
    # x = Dense(10, name='dense3')(x)
    # x = Activation(activation='sigmoid', name='act3')(x)
    # output layer
    x = Dense(nclass, name='pre_activation')(x)
    y_pred = Activation(activation='tanh', name='output')(x)
    # append MFoM layers
    y_mfom, out_mfom = join_mfom(tip_layer=y_pred, nclass=nclass)

    m_model = Model(input=[y_mfom, feat_input], output=out_mfom)
    m_model.compile(loss=obj.mfom_macrof1, optimizer='Adam')
    m_model.summary()
    return m_model


def join_mfom(tip_layer, nclass):
    # === MFoM head ===
    # misclassification layer, feed Y
    y_mfom = Input(shape=(nclass,), name='y_true')
    psi = kmfom.UvZMisclassification(name='uvz_misclass')([y_mfom, tip_layer])
    # class Loss function layer
    # NOTE: you may want to add regularization or constraints
    out_mfom = kmfom.SmoothErrorCounter(name='smooth_error_counter',
                                        # alpha_constraint=constraints.min_max_norm(min_value=-4., max_value=4.),
                                        alpha_regularizer=regs.l1(0.001),
                                        # beta_constraint=constraints.min_max_norm(min_value=-4., max_value=4.),
                                        beta_regularizer=regs.l1(0.001)
                                        )(psi)
    return y_mfom, out_mfom


def cut_mfom(model):
    # calc accuracy: cut MFoM head, up to sigmoid output
    input = model.get_layer(name='main_input').output
    out = model.get_layer(name='output').output
    cut_net = Model(input=input, output=out)
    return cut_net


def generate_dataset(n_smp=300, ratio=0.3, n_feat=2, n_cls=2, multilabel=True):
    if multilabel:
        x, y = make_multilabel_classification(n_samples=n_smp, n_features=n_feat,
                                              n_classes=n_cls, n_labels=1,
                                              allow_unlabeled=False,
                                              random_state=RANDOM_SEED)
    else:
        x, y = sk.datasets.make_blobs(n_samples=n_smp,
                                      centers=n_cls,
                                      n_features=n_feat,
                                      random_state=RANDOM_SEED)
        le = MultiLabelBinarizer()
        y = le.fit_transform(y[:, np.newaxis])
    # scaler = sk.preprocessing.StandardScaler()
    # x = scaler.fit_transform(x)
    x_tr, x_tst, y_tr, y_tst = train_test_split(x, y, test_size=ratio, random_state=RANDOM_SEED)
    return x_tr, x_tst, y_tr, y_tst


if __name__ == '__main__':
    dim = 20
    nclass = 5

    # multi-label dataset
    x_train, x_test, y_train, y_test = generate_dataset(n_smp=1000, n_feat=dim,
                                                        n_cls=nclass, multilabel=False)
    mask = y_train.sum(axis=-1) != nclass
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = y_test.sum(axis=-1) != nclass
    x_test = x_test[mask]
    y_test = y_test[mask]

    # baseline model architecture
    b_model = baseline_model(feat_dim=dim, nclass=nclass)
    # mfom model
    m_model = mfom_model(feat_dim=dim, nclass=nclass)

    # baseline training
    hist_base = b_model.fit(x_train, y_train, nb_epoch=30, batch_size=8)
    # mfom training
    hist_mfom = m_model.fit([y_train, x_train], y_train, nb_epoch=30, batch_size=8)

    # === baseline evaluation ===
    y_pred = b_model.predict(x_test)
    # evaluate
    eer_val = MT.class_wise_eer(y_true=y_test, y_pred=y_pred)
    print('Baseline EER: %.4f' % np.mean(eer_val))
    # plt.plot(hist_base.history['loss'])
    pltr.show_decision_boundary(b_model, X=x_test, Y=y_test)

    # === mfom evaluation ===
    cut_model = cut_mfom(m_model)
    y_pred = cut_model.predict(x_test)
    # evaluate
    eer_val = MT.class_wise_eer(y_true=y_test, y_pred=y_pred)
    print('MFoM EER: %.4f' % np.mean(eer_val))

    # history plot, alpha and beta params of MFoM
    m = m_model.get_layer('smooth_error_counter')
    print('alpha: ', K.get_value(m.alpha))
    print('beta: ', K.get_value(m.beta))
    pltr.show_decision_boundary(cut_model, X=x_test, Y=y_test)
    # plt.plot(hist_mfom.history['loss'])
    plt.show()
