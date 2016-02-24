import matplotlib.pyplot as plt
import numpy as np
from utils import mnist_loader
from ann import network
from ann import rnd_feedback_network


def plotter(data, xlim=None, ylim=None, title=None, xlab=None, ylab=None, fname=None):
    # plot settings
    plt.clf()  # Clear the current figure (prevents multiple labels)
    labelfont = {
        'family': 'sans-serif',  # (cursive, fantasy, monospace, serif)
        'color': 'black',  # html hex or colour name
        'weight': 'normal',  # (normal, bold, bolder, lighter)
        'size': 14,  # default value:12
    }
    titlefont = {
        'family': 'serif',
        'color': 'black',
        'weight': 'bold',
        'size': 16,
    }

    plt.plot(data)
    axes = plt.gca()
    if xlim != None:
        axes.set_xlim(xlim)  # x-axis bounds
    if ylim != None:
        axes.set_ylim(ylim)  # y-axis bounds
    if title != None:
        plt.title(title, fontdict=titlefont)
    if xlab != None:
        plt.xlabel(xlab, fontdict=labelfont)
    if ylab != None:
        plt.ylabel(ylab, fontdict=labelfont)
    if fname != None:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()


def plotter_std(mean_data, std_data, xlim=None, ylim=None, title=None, xlab=None, ylab=None, fname=None):
    # plot settings
    plt.clf()  # Clear the current figure (prevents multiple labels)
    labelfont = {
        'family': 'sans-serif',  # (cursive, fantasy, monospace, serif)
        'color': 'black',  # html hex or colour name
        'weight': 'normal',  # (normal, bold, bolder, lighter)
        'size': 14,  # default value:12
    }
    titlefont = {
        'family': 'serif',
        'color': 'black',
        'weight': 'bold',
        'size': 16,
    }

    plt.errorbar(list(range(1, len(mean_data) + 1)), np.array(mean_data),
                 np.array(std_data), linestyle='None', marker='o')
    axes = plt.gca()
    if xlim != None:
        axes.set_xlim(xlim)  # x-axis bounds
    if ylim != None:
        axes.set_ylim(ylim)  # y-axis bounds
    if title != None:
        plt.title(title, fontdict=titlefont)
    if xlab != None:
        plt.xlabel(xlab, fontdict=labelfont)
    if ylab != None:
        plt.ylabel(ylab, fontdict=labelfont)
    if fname != None:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print("MNIST data is loaded...")

exp_path = 'data/experiment/'
nist_exp_path = 'data/experiment/nist/'
num_exp = 10
epochs = 50
mini_batch = 10
learn_rate = 3.0

exp = 'nist_rfa'

" " " MSE network with sigmoid output layer " " "
if exp == 'nist_mse':
    all_loss = []
    all_eval = []
    for i in xrange(0, num_exp):
        print('Experiment number: {0}'.format(i))
        net = network.Network([784, 30, 10])
        eval, loss = net.SGD(training_data, epochs, mini_batch, learn_rate, test_data=test_data)
        all_loss.append(loss)
        all_eval.append(eval)
    loss_mean = np.mean(all_loss, axis=0)
    loss_std = np.std(all_loss, axis=0)
    eval_mean = np.mean(all_eval, axis=0)
    eval_std = np.std(all_eval, axis=0)
    print(loss_mean)
    print(eval_mean)

    # plot evaluation result
    plotter_std(eval_mean, eval_std, xlim=[0, len(eval)], ylim=[7000, 10000],
            title='Evaluation results: 784-30-10',
            xlab='# of epoch',
            ylab='# of correct recognition',
            fname=nist_exp_path + exp + '_eval_progress.png')

    # plot loss progress
    plotter_std(loss_mean, loss_std, xlim=[0, len(loss)], ylim=[0, 15],
            title='MSE loss function: 784-30-10',
            xlab='# of epoch',
            ylab='MSE value',
            fname=nist_exp_path + exp + '_loss_progress.png')


" " " random feedback alignment experiment " " "
if exp == 'nist_rfa':
    all_loss = []  # [[13, 10, 9, 7, 5], [12, 10, 7, 5, 3], [11, 8, 7, 5, 3]]
    all_eval = []  # [[9000, 9210, 9302, 9490, 9500], [9100, 9210, 9202, 9450, 9600], [9100, 9140, 9252, 9470, 9560]]
    for i in xrange(0, num_exp):
        print('Experiment number: {0}'.format(i))
        net = rnd_feedback_network.RFANetwork([784, 30, 10])
        eval, loss = net.SGD(training_data, epochs, mini_batch, learn_rate, test_data=test_data)
        all_loss.append(loss)
        all_eval.append(eval)
    loss_mean = np.mean(all_loss, axis=0)
    loss_std = np.std(all_loss, axis=0)
    eval_mean = np.mean(all_eval, axis=0)
    eval_std = np.std(all_eval, axis=0)
    print(loss_mean)
    print(eval_mean)

    # plot evaluation result
    plotter_std(eval_mean, eval_std, xlim=[0, len(eval_mean)], ylim=[7000, 10000],
                title='Random feedback alignment, evaluation results: 784-30-10',
                xlab='# of epoch',
                ylab='# of correct recognition',
                fname=nist_exp_path + exp + '_eval_progress.png')

    # plot loss progress
    plotter_std(loss_mean, loss_std, xlim=[0, len(loss_mean)], ylim=[0, 15],
                title='Random feedback alignment, MSE loss function: 784-30-10',
                xlab='# of epoch',
                ylab='MSE value',
                fname=nist_exp_path + exp + '_loss_progress.png')





# microF1 metric
# m = sk.f1_score(y_true=, y_pred=, average='micro')



