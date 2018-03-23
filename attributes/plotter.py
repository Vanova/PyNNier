import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('agg')
plt.style.use('seaborn')


def plot_stackbars(data, bar_labels, axis_labels, file_name):
    df = pd.DataFrame(data, columns=bar_labels)
    df.plot(kind='barh', stacked=True)
    plt.yticks(range(len(axis_labels)), axis_labels)
    plt.xlim([0, 1.05])
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.show()
    plt.savefig(file_name, bbox_inches="tight")