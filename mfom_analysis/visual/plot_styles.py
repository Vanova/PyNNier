import numpy as np

# plot settings
ticks_font = {
    'family': 'sans-serif',  # (cursive, fantasy, monospace, serif)
    'color': 'black',  # html hex or colour name
    'weight': 'normal',  # (normal, bold, bolder, lighter)
    'size': 7,  # default value:12
}
label_font = {
    'family': 'sans-serif',  # (cursive, fantasy, monospace, serif)
    'color': 'black',  # html hex or colour name
    'weight': 'normal',  # (normal, bold, bolder, lighter)
    'size': 14,  # default value:12
}
title_font = {
    'family': 'serif',
    'color': 'black',
    'weight': 'bold',
    'size': 10,
}


def retrieve_n_class_color_cubic(N):
    """
    Retrieve color code for N given classes
    Input: class number
    Output: list of RGB color code
    """

    # manualy encode the top 8 colors
    # the order is intuitive to be used
    color_list = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (0, 0, 0),
        (1, 1, 1)
    ]

    # if N is larger than 8 iteratively generate more random colors
    np.random.seed(1)  # pre-define the seed for consistency

    interval = 0.5
    while len(color_list) < N:
        the_list = []
        iterator = np.arange(0, 1.0001, interval)
        for i in iterator:
            for j in iterator:
                for k in iterator:
                    if (i, j, k) not in color_list:
                        the_list.append((i, j, k))
        the_list = list(set(the_list))
        np.random.shuffle(the_list)
        color_list.extend(the_list)
        interval = interval / 2.0

    return color_list[:N]
