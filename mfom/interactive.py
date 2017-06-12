from matplotlib.widgets import Slider, Button
from multiprocessing.pool import ThreadPool
import mfom.visual.plotter as mfom_plt
import matplotlib.pyplot as plt
import mfom.utils.toy_scores as mfom_toy
import mfom.utils.dcase_scores as mfom_dcase
import cost_function as mfom_cost
from mfom.utils.metrics import eer, sklearn_rocch, sklearn_pav
import sklearn.metrics as sk_metrics
import numpy as np



# TODO single interactive view template but can add different views
# scatters(a, b), hist(a, b), ROC, ROCCH

class InteractiveSliderView(object):
    def __init__(self, fig, data_canvas, init_a, init_b, views_to_track=None):
        self.fig = fig
        self.data_canvas = data_canvas
        # Sliders
        axis_color = 'lightgoldenrodyellow'
        a_slider_ax = self.fig.add_axes([0.1, 0.15, 0.65, 0.03], facecolor=axis_color)
        self.a_slider = Slider(a_slider_ax, 'Alpha', 0.1, 30.0, valinit=init_a)
        b_slider_ax = self.fig.add_axes([0.1, 0.1, 0.65, 0.03], facecolor=axis_color)
        self.b_slider = Slider(b_slider_ax, 'Beta', -0.5, 30.0, valinit=init_b)

        # Bind slider with parameters' values
        self.a_slider.on_changed(self.sliders_on_changed)
        self.b_slider.on_changed(self.sliders_on_changed)

        # Reset button
        reset_button_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        self.reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.9')
        self.reset_button.on_clicked(self.reset_button_on_clicked)

    def sliders_on_changed(self, val):
        # recalculate data in the new Thread
        new_a = self.a_slider.val
        new_b = self.b_slider.val
        # update canvas
        self.data_canvas.update(new_a, new_b)
        # redraw main figure
        self.fig.canvas.draw_idle()

    def reset_button_on_clicked(self, mouse_event):
        self.a_slider.reset()
        self.b_slider.reset()


class InteractiveErrorCounterView(object):

    def __init__(self, ax, y_true, y_score, idata):
        self.ax = ax
        self.y_true = y_true
        self.y_score = y_score
        self.idata = idata
        # calculate ROC
        _, _, fpr, fnr, thresholds = mfom_toy.count_discrete_errors(y_true, y_score)
        eer_val = eer(y_true, y_score)
        # plot FNR/FPR distributions
        mfom_plt.view_fnr_fpr_dist(ax, fnr, fpr, thresholds, eer_val)

    def update(self, new_a, new_b):
        # recalculate the data
        self.idata.update(new_a, new_b)
        # process scores
        y_score_df, y_true_df = mfom_toy.pool_scores(p_df=self.idata.mfom_P, y_df=self.idata.orig_Y)
        # clean up the hist plot
        self.ax.cla()
        # redraw
        _, _, fpr, fnr, thresholds = mfom_toy.count_discrete_errors(y_true_df, y_score_df)
        eer_val = eer(y_true_df, y_score_df)
        # plot FNR/FPR distributions
        mfom_plt.view_fnr_fpr_dist(ax, fnr, fpr, thresholds, eer_val)



class InteractiveHistogramView(object):
    """
    Return view of histogram on axes
    tar_pool: 1D array
    ntar_pool: 1D array
    gaus_fit: fit histogram with Gaussian
    ax: plot axes
    """
    def __init__(self, ax, tar, nontar, inter_data, bins=5, gaus_fit=True, min=0., max=1.):
        self.idata = inter_data
        self.ax = ax
        self.min = min
        self.max = max
        self.bins = bins
        mfom_plt.view_histogram(self.ax, tar, nontar, bins=self.bins, min=self.min, max=self.max)

    def update(self, new_a, new_b):
        # recalculate the data
        p, y = self.idata.update(new_a, new_b)
        tar, ntar = mfom_toy.pool_split_tnt(p_df=p, y_df=y)
        # clean up the hist plot
        self.ax.cla()
        # redraw
        mfom_plt.view_histogram(self.ax, tar, ntar, bins=self.bins, min=self.min, max=self.max)


class MFoMInteractiveData(object):
    def __init__(self, orig_y_true, orig_y_score, def_alpha, def_beta):
        self.pool = ThreadPool(processes=1)
        self.orig_Y = orig_y_true
        self.orig_P = orig_y_score
        loss_scores = mfom_cost._uvz_loss_scores(y_true=orig_y_true.values, y_pred=orig_y_score.values,
                                                 alpha=def_alpha, beta=def_beta)
        self.mfom_P = mfom_toy.arr2DataFrame(1. - loss_scores, row_id=orig_y_score.index, col_id=orig_y_score.columns)

    def update(self, a, b):
        # did not notice the difference with Thread :(
        async_result = self.pool.apply_async(mfom_cost._uvz_loss_scores, (self.orig_Y.values, self.orig_P.values, a, b, True))
        loss_scores = async_result.get()
        # loss_scores = mfom_cost._uvz_loss_scores(self.orig_Y.values, self.orig_P.values, a, b, True)
        self.mfom_P = mfom_toy.arr2DataFrame(1. - loss_scores, row_id=self.orig_P.index, col_id=self.orig_P.columns)
        return self.mfom_P, self.orig_Y


if __name__ == '__main__':
    P_df = mfom_dcase.read_dcase('data/test_scores/results_fold1.txt')
    Y_df = mfom_dcase.read_dcase('data/test_scores/y_true_fold1.txt')

    # pooled MFoM scores
    def_alpha = 1.
    def_beta = 0.
    intr_data = MFoMInteractiveData(Y_df, P_df, def_alpha, def_beta)

    # ===
    # Histograms: MFoM scores
    # ===
    # plot interactive histogram
    # tar, ntar = mfom_toy.pool_split_tnt(p_df=intr_data.mfom_P, y_df=intr_data.orig_Y)
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # ax = fig.add_subplot(1, 1, 1, frameon=True)
    # fig.subplots_adjust(left=0.1, bottom=0.25)
    #
    # hist = InteractiveHistogramView(ax, tar, ntar, intr_data, bins=10)
    # vis = InteractiveSliderView(fig, data_canvas=hist, init_a=def_alpha, init_b=def_beta)
    # plt.show()

    # ===
    # FNR vs FPR: MFoM scores
    # ===
    y_score_df, y_true_df = mfom_toy.pool_scores(p_df=intr_data.mfom_P, y_df=intr_data.orig_Y)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    fig.subplots_adjust(left=0.1, bottom=0.30)

    er_cnt = InteractiveErrorCounterView(ax, y_true=y_true_df, y_score=y_score_df, idata=intr_data)
    vis = InteractiveSliderView(fig, data_canvas=er_cnt, init_a=def_alpha, init_b=def_beta)
    plt.show()

