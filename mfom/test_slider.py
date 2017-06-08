from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


def score_data(amp, freq):
    """
    Data function
    """
    t = np.arange(0.0, 1.0, 0.001)
    return amp * sin(2 * pi * freq * t)


# TODO single interactive view template but can add different views
# scatters(a, b), hist(a, b), ROC, ROCCH
class InteractiveSliderView(object):
    def __init__(self, t, data, xmin=0, xmax=1, ymin=-10, ymax=10, views_to_track=None):
        self.fig = plt.figure()
        # Draw the plot
        ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.1, bottom=0.25)
        # [line] = ax.plot(t, score_data(alpha, beta), linewidth=2, color='red')
        [self.data_canvas] = ax.plot(t, data, linewidth=2, color='red')

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        # Sliders
        axis_color = 'lightgoldenrodyellow'
        a_slider_ax = self.fig.add_axes([0.1, 0.15, 0.65, 0.03], axisbg=axis_color)
        self.a_slider = Slider(a_slider_ax, 'Alpha', 0.1, 10.0, valinit=alpha)
        b_slider_ax = self.fig.add_axes([0.1, 0.1, 0.65, 0.03], axisbg=axis_color)
        self.b_slider = Slider(b_slider_ax, 'Beta', 0.1, 30.0, valinit=beta)

        # Bind slider with parameters' values
        self.a_slider.on_changed(self.sliders_on_changed)
        self.b_slider.on_changed(self.sliders_on_changed)

        # Reset button
        reset_button_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
        reset_button.on_clicked(self.reset_button_on_clicked)
        plt.show()
        # return data_canvas, a_slider, b_slider, fig

    def sliders_on_changed(self, val):
        self.data_canvas.set_ydata(score_data(self.a_slider.val, self.b_slider.val))
        self.fig.canvas.draw_idle()

    def reset_button_on_clicked(self, mouse_event):
        self.a_slider.reset()
        self.b_slider.reset()


if __name__ == '__main__':
    alpha = 5
    beta = 3
    t = np.arange(0.0, 1.0, 0.001)
    data = score_data(alpha, beta)
    vis = InteractiveSliderView(t, data)
