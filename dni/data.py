import numpy as np
np.random.seed(1)


def generate_dataset(output_dim=8, num_examples=1000):
    """Dataset: sum of 2 binary digits"""

    def int2vec(x, dim=output_dim):
        out = np.zeros(dim)
        binrep = np.array(list(np.binary_repr(x))).astype('int')
        out[-len(binrep):] = binrep
        return out

    x_left_int = (np.random.rand(num_examples) * 2 ** (output_dim - 1)).astype('int')
    x_right_int = (np.random.rand(num_examples) * 2 ** (output_dim - 1)).astype('int')
    y_int = x_left_int + x_right_int

    x = list()
    for i in range(len(x_left_int)):
        x.append(np.concatenate((int2vec(x_left_int[i]), int2vec(x_right_int[i]))))

    y = list()
    for i in range(len(y_int)):
        y.append(int2vec(y_int[i]))

    x = np.array(x)
    y = np.array(y)
    return x, y
