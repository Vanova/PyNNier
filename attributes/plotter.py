import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('agg')
plt.style.use('seaborn')


def plot_stackbars(data, bar_labels, axis_labels, file_name):
    df = pd.DataFrame(data, columns=bar_labels)
    df.plot(kind='barh', stacked=True)
    plt.yticks(range(len(axis_labels)), axis_labels, fontsize=12)
    plt.xlim([0, 1.05])
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, fontsize=12)
    plt.show()
    plt.savefig(file_name, bbox_inches="tight")


def plot_chord():
    pass


if __name__ == '__main__':
    """
    see source: https://bokeh.pydata.org/en/latest/docs/user_guide/plotting.html
    """
    from numpy import linspace
    from scipy.stats.kde import gaussian_kde

    from bokeh.io import output_file, show
    from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter
    from bokeh.plotting import figure
    from bokeh.sampledata.perceptions import probly

    import colorcet as cc

    output_file("joyplot.html")


    def joy(category, data, scale=20):
        return list(zip([category] * len(data), scale * data))


    cats = list(reversed(probly.keys()))

    palette = [cc.rainbow[i * 15] for i in range(17)]

    x = linspace(-20, 110, 500)

    source = ColumnDataSource(data=dict(x=x))

    p = figure(y_range=cats, plot_width=700, x_range=(-5, 105), toolbar_location=None)

    for i, cat in enumerate(reversed([cats[0]])):
        pdf = gaussian_kde(probly[cat])
        y = joy(cat, pdf(x))
        source.add(y, cat)
        p.patch('x', cat, color=palette[i], alpha=0.6, line_color="black", source=source)

        source2 = ColumnDataSource(data=dict(x=x))
        pdf = gaussian_kde(probly[cats[3]])
        y2 = joy(cat, pdf(x), scale=30)
        source2.add(y2, cat)
        p.patch('x', cat, color=palette[(2*i) % 17], alpha=0.3, line_color="black", source=source2)

        # x = [1, 3, 2]
        # y = [(cat, 3),
        #      (cat, 2),
        #      (cat, 1)]
        # p.patch(x, y, color=palette[8], alpha=0.4, line_color="black")

    p.outline_line_color = None
    p.background_fill_color = "#efefef"

    p.xaxis.ticker = FixedTicker(ticks=list(range(0, 101, 10)))
    p.xaxis.formatter = PrintfTickFormatter(format="%d%%")

    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = "#dddddd"
    p.xgrid.ticker = p.xaxis[0].ticker

    p.axis.minor_tick_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.axis_line_color = None

    p.y_range.range_padding = 0.12

    show(p)