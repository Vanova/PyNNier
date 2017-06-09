from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from multiprocessing.pool import ThreadPool
import mfom.visual.plotter as mfom_plt
from mfom.utils import toy_scores as TS
import cost_function as mfom_cost
import mfom.utils.dcase_scores as mfom_dcase
from scipy.stats import norm


# TODO single interactive view template but can add different views
# scatters(a, b), hist(a, b), ROC, ROCCH

class InteractiveSliderView(object):
    def __init__(self, fig, data_canvas, init_a, init_b, views_to_track=None):
        self.data_canvas = data_canvas

        self.fig = fig
        # self.data_canvas = data_canvas
        # Sliders
        axis_color = 'lightgoldenrodyellow'
        a_slider_ax = self.fig.add_axes([0.1, 0.15, 0.65, 0.03], axisbg=axis_color)
        self.a_slider = Slider(a_slider_ax, 'Alpha', 0.1, 30.0, valinit=init_a)
        b_slider_ax = self.fig.add_axes([0.1, 0.1, 0.65, 0.03], axisbg=axis_color)
        self.b_slider = Slider(b_slider_ax, 'Beta', -0.5, 30.0, valinit=init_b)

        # Bind slider with parameters' values
        self.a_slider.on_changed(self.sliders_on_changed)
        self.b_slider.on_changed(self.sliders_on_changed)

        # Reset button
        reset_button_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.9')
        reset_button.on_clicked(self.reset_button_on_clicked)

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
        mfom_plt.view_histogram(self.ax, tar, ntar, bins=self.bins, min=self.min, max=self.max)

    def update(self, new_a, new_b):
        # recalculate the data
        self.idata.update(new_a, new_b)
        tar, ntar = TS.pool_split_tnt(p_df=self.idata.ls_df, y_df=self.idata.orig_Y)
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
        self.ls_df = TS.arr2DataFrame(1. - loss_scores, row_id=orig_y_score.index, col_id=orig_y_score.columns)

    def update(self, a, b):
        # did not notice the difference with Thread :(
        async_result = self.pool.apply_async(mfom_cost._uvz_loss_scores, (self.orig_Y.values, self.orig_P.values, a, b, True))
        loss_scores = async_result.get()
        # loss_scores = mfom_cost._uvz_loss_scores(self.orig_Y.values, self.orig_P.values, a, b, True)
        self.ls_df = TS.arr2DataFrame(1. - loss_scores, row_id=self.orig_P.index, col_id=self.orig_P.columns)
        return self.ls_df, self.orig_Y


if __name__ == '__main__':
    P_df = mfom_dcase.read_dcase('data/test_scores/results_fold1.txt')
    Y_df = mfom_dcase.read_dcase('data/test_scores/y_true_fold1.txt')

    # ===
    # Histograms: MFoM scores
    # ===
    # pooled scores
    def_alpha = 1.
    def_beta = 0.
    inter_data = MFoMInteractiveData(Y_df, P_df, def_alpha, def_beta)
    tar, ntar = TS.pool_split_tnt(p_df=inter_data.ls_df, y_df=inter_data.orig_Y)

    # plot interactive histogram
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    fig.subplots_adjust(left=0.1, bottom=0.25)

    hist = InteractiveHistogramView(ax, tar, ntar, inter_data, bins=10)
    vis = InteractiveSliderView(fig, data_canvas=hist, init_a=def_alpha, init_b=def_beta)
    plt.show()

