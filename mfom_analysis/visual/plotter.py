import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import cm
import numpy as np
from scipy.stats import norm
import plot_styles as plts

plt.style.use('seaborn')


def colored_table(ax, vals, row_lab, col_lab):
    """
    Plot colored table
    ax: axis of fig.add_subplot(...)
    vals: 2D array, numpy.ndarray
    row_lab: list of row indices
    col_lab: list of column indices
    """
    normal = mpl.colors.Normalize(vmin=vals.min() - .2, vmax=vals.max() + .2)
    ax.table(cellText=vals, rowLabels=row_lab, colLabels=col_lab,
             colWidths=[0.03] * vals.shape[1],
             cellColours=plt.cm.coolwarm(normal(vals)), alpha=0.5, loc='left')
    return ax


def view_histogram(ax, tar, nontar, bins=5, gaus_fit=True, min=0., max=1.):
    """
    Return view of histogram on axes
    tar_pool: 1D array
    ntar_pool: 1D array
    gaus_fit: fit histogram with Gaussian
    ax: plot axes
    """
    ax.hist(tar, label='Target', bins=bins, alpha=0.5, color='b')
    ax.hist(nontar, label='Non-target', bins=bins, alpha=0.5, color='g')
    ax.legend(loc='upper right')

    if gaus_fit:
        # target scores 'best fit' line
        x = np.linspace(min, max, 100)
        (mu, s) = norm.fit(tar)
        y = norm.pdf(x, mu, s)
        ax.plot(x, y, alpha=0.3, color='b')

        # non-target scores 'best fit' line
        (mu, s) = norm.fit(nontar)
        y = norm.pdf(x, mu, s)
        ax.plot(x, y, '--', alpha=0.3, color='g')
    return ax


def view_roc_curve(ax, fpr, tpr, label=None, eer_val=None, roc_auc=None, color=None, title=''):
    lw = 0.5
    if label is None:
        label = 'ROC (area = %0.2f)' % roc_auc
    ax.plot(fpr, tpr, marker='o', markersize=3, linestyle='--', color=color, label=label)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.plot(eer_val, 1. - eer_val, linestyle=' ', marker='*', markersize=10, label='EER = %0.2f' % eer_val, color='red')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontdict=plts.label_font)
    ax.set_ylabel('True Positive Rate', fontdict=plts.label_font)
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def view_fnr_fpr_dist(ax, fnr, fpr, thresholds, eer_val):
    lw = 1
    ax.plot(thresholds, fpr, marker='o', linestyle='--', label='FPR')
    ax.plot(thresholds, fnr, marker='s', linestyle='--', label='FNR')
    ax.plot(thresholds, np.abs(fnr - fpr), marker='^', linestyle=':', alpha=0.8, lw=lw, label='|FNR - FPR|')
    ax.plot(thresholds, np.abs(fnr + fpr), marker='.', linestyle=':', alpha=0.8, lw=lw, label='FNR + FPR')

    id_x = np.argmin(np.abs(fnr - fpr))
    ax.plot(thresholds[id_x], eer_val, linestyle=' ', marker='*', markersize=15, label='EER = %0.2f' % eer_val,
            color='red')
    ax.set_xlabel('Thresholds', fontdict=plts.label_font)
    ax.set_ylabel('Error rate', fontdict=plts.label_font)
    ax.legend(loc='best')
    return ax


def surface_view(ax, x, y, z, xlim, ylim, labels):
    surf = ax.plot_surface(x, y, z,
                           rstride=1, cstride=1, linewidth=0,
                           alpha=0.5, cmap=cm.coolwarm, antialiased=True, zorder=0)
    ax.set_xlabel(labels[0], fontdict=plts.label_font)
    ax.set_ylabel(labels[1], fontdict=plts.label_font)
    ax.set_zlabel(labels[2], fontdict=plts.label_font)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return surf


def show_curves(y, legend=None, labels=None, title=None):
    """
    Plot curve, starting from x = 0
    """
    for it in y:
        plt.plot(it)
    if legend:
        plt.legend(legend, loc='best')
    if labels:
        plt.xlabel(labels[0], fontdict=plts.label_font)
        plt.ylabel(labels[1], fontdict=plts.label_font)
    if title:
        plt.title(title, fontdict=plts.title_font)
    plt.grid()
    plt.show()


def show_decision_boundary(model, dim_red_method='pca',
                           X=None, Y=None,
                           xrg=None, yrg=None,
                           Nx=300, Ny=300,
                           scatter_sample=None,
                           figsize=[6, 6], alpha=0.7,
                           random_state=111):
    """
    Plot decision boundary for any two dimension classification models
        in sklearn.

    Input:
        model: sklearn classification model class - already fitted
                (with "predict" and "predict_proba" method)

        dim_red_method: sklearn dimension reduction model
                (with "fit_transform" and "inverse_transform" method)

        X (nparray): dataset to project over decision boundary (X)
        Y (nparray): ndarray, binary labels

        xrg (list/tuple): xrange
        yrg (list/tuple): yrange
        Nx (int): x axis grid size
        Ny (int): y axis grid size
        figsize, alpha are parameters in matplotlib

    Output:
        matplotlib figure object
    """
    try:
        getattr(model, 'predict')
    except:
        print("model do not have method predict 'predict' ")
        return None

    use_prob = True
    try:
        getattr(model, 'predict_proba')
    except:
        print("model do not have method predict 'predict_proba' ")
        use_prob = False

    # convert X into 2D data
    dr_model = None
    scaler = None
    if X is not None:
        if X.shape[1] == 2:
            X2D = X
        elif X.shape[1] > 2:
            # leverage PCA to dimension reduction to 2D if not already
            scaler = StandardScaler()
            if dim_red_method == 'pca':
                dr_model = PCA(n_components=2)
            elif dim_red_method == 'kernal_pca':
                dr_model = KernelPCA(n_components=2,
                                     fit_inverse_transform=True)
            else:
                print('dim_red_method {0} is not supported'.format(
                    dim_red_method))

            X2D = dr_model.fit_transform(scaler.fit_transform(X))
        else:
            print('X dimension is strange: {0}'.format(X.shape))
            return None

        # extract two dimension info.
        x1 = X2D[:, 0].min() - 0.1 * (X2D[:, 0].max() - X2D[:, 0].min())
        x2 = X2D[:, 0].max() + 0.1 * (X2D[:, 0].max() - X2D[:, 0].min())
        y1 = X2D[:, 1].min() - 0.1 * (X2D[:, 1].max() - X2D[:, 1].min())
        y2 = X2D[:, 1].max() + 0.1 * (X2D[:, 1].max() - X2D[:, 1].min())

    # inti xrg and yrg based on given value
    if xrg is None:
        if X is None:
            xrg = [-10, 10]
        else:
            xrg = [x1, x2]

    if yrg is None:
        if Y is None:
            yrg = [-10, 10]
        else:
            yrg = [y1, y2]

    # generate grid, mesh, and X for model prediction
    xgrid = np.arange(xrg[0], xrg[1], 1. * (xrg[1] - xrg[0]) / Nx)
    ygrid = np.arange(yrg[0], yrg[1], 1. * (yrg[1] - yrg[0]) / Ny)

    xx, yy = np.meshgrid(xgrid, ygrid)

    # initialize figure & axes object
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # get data from model predictions
    X_full_grid = np.c_[xx.ravel(), yy.ravel()]
    if dr_model:
        X_full_grid = scaler.inverse_transform(
            dr_model.inverse_transform(X_full_grid))

    if use_prob:
        Ypp = model.predict_proba(X_full_grid)
        Yp = model.predict(X_full_grid)  # labels in sklearn y_pred_grid
    else:
        Ypp = model.predict(X_full_grid)
        if (Ypp < 0).any():
            Ypp = (Ypp + 1.) * 0.5
        Yp = Ypp > 0.5

    nclass = Y.shape[1]
    colors = np.array(plts.retrieve_n_class_color_cubic(N=nclass))

    # get decision boundary line
    Yp_single = binary_to_labels(Yp)
    Yp_single = Yp_single.reshape(xx.shape)
    Yb = np.zeros(xx.shape)

    Yb[:-1, :] = np.maximum((Yp_single[:-1, :] != Yp_single[1:, :]), Yp_single[:-1, :])
    Yb[1:, :] = np.maximum((Yp_single[:-1, :] != Yp_single[1:, :]), Yp_single[1:, :])
    Yb[:, :-1] = np.maximum((Yp_single[:, :-1] != Yp_single[:, 1:]), Yp_single[:, :-1])
    Yb[:, 1:] = np.maximum((Yp_single[:, :-1] != Yp_single[:, 1:]), Yp_single[:, 1:])

    # plot decision boundary first
    ax.imshow(Yb, origin='lower', interpolation=None, cmap='Greys',
              extent=[xrg[0], xrg[1], yrg[0], yrg[1]],
              alpha=1.0)

    # plot probability surface
    zz = np.dot(Ypp, colors)
    zz_r = zz.reshape(xx.shape[0], xx.shape[1], 3)
    ax.imshow(zz_r, origin='lower', interpolation=None,
              extent=[xrg[0], xrg[1], yrg[0], yrg[1]],
              alpha=alpha)

    # add scatter plot for X & Y if given
    if X is not None:
        # down sample point if needed
        if Y is not None:
            if scatter_sample is not None:
                X2DS, _, YS, _ = train_test_split(X2D, Y, stratify=Y,
                                                  train_size=scatter_sample,
                                                  random_state=random_state)
            else:
                X2DS = X2D
                YS = Y
        else:
            if scatter_sample is not None:
                X2DS, _ = train_test_split(X2D, train_size=scatter_sample,
                                           random_state=random_state)
            else:
                X2DS = X2D

        # convert Y into point color
        if Y is not None:
            cYS = np.dot(YS, colors) # TODO check dim
            # cYS = np.clip(cYS, a_min=0., a_max=1.)

        if Y is not None:
            ax.scatter(X2DS[:, 0], X2DS[:, 1], c=cYS, s=20, edgecolor='k')
        else:
            ax.scatter(X2DS[:, 0], X2DS[:, 1])

    # add legend on each class
    # colors_bar = []
    # for v1 in colors:
    #     v1 = list(v1)
    #     v1.append(alpha)
    #     colors_bar.append(v1)
    #
    # # create a patch (proxy artist) for every color
    # patches = [mpatches.Patch(color=colors_bar[i],
    #                           label="Class {k}".format(k=i))
    #            for i in range(nclass)]
    # # put those patched as legend-handles into the legend
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),
    #            loc=2, borderaxespad=0., framealpha=0.5)

    # make the figure nicer
    ax.set_title('Classification decision boundary')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(xrg)
    ax.set_ylim(yrg)
    ax.set_xticks(np.arange(xrg[0], xrg[1], (xrg[1] - xrg[0]) / 5.))
    ax.set_yticks(np.arange(yrg[0], yrg[1], (yrg[1] - yrg[0]) / 5.))
    ax.grid(False)
    return fig, scaler, dr_model


def binary_to_labels(bin_labs):
    str_bin = [str(b) for b in bin_labs]
    le = LabelEncoder()
    dig_labs = le.fit_transform(str_bin)
    return dig_labs
