"""
Speech attributes content analysis in NIST LRE 2017 dataset
"""
import os
import time
import pickle
import collections
import os.path as path
import numpy as np
import metrics.metrics as metr
import utils.kaldi.io as kio
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
import plotter as pltr
from scipy.optimize import curve_fit

MANNER_CLS = []
PLACE_CLS = []
ATTRIBUTES_CLS = {'manner': ['fricative', 'glides', 'nasal', 'other', 'silence', 'stop', 'voiced', 'vowel'],
                  'place': ['coronal', 'dental', 'glottal', 'high', 'labial',
                            'low', 'mid', 'other', 'palatal', 'silence', 'velar'],
                  'fusion_manner': [],
                  'fusion_place': []}
N_HIST = 100


class Stats(object):
    num = 'number'
    len = 'length'
    mean_mean = 'mean_mean'
    gmean = 'global_mean'
    cont_mean = 'continues_mean'
    hist = 'histogram'


def scan_folder(lang_path, attrib_cls):  # TODO scan by name tamplate!!!
    flist = []
    for j in xrange(n_jobs):
        p = '%s/res/%s/scores.%i.txt' % (lang_path, attrib_cls, j + 1)
        flist.append(p)
    return flist


def utterance_number(file_name):
    """
    Calculate number of utterances in file storage by start sign (e.g., '[')
    :param file_name: storage txt file, with ARK files
    """
    match = kio.ArkReader.grep(file_name, kio.START_ARK_MARK)
    return len(match)


def utterance_length(file_name, window_sz=20, shift_sz=10, sample_rate=8000):
    """
    Calculate total hours in the file_name storage of ARK files
    :param file_name: storage of several ARK files
    :param window_sz: size of frame in msec, e.g. 20ms
    :param shift_sz: size of overlapping shift in msec, e.g. 10ms
    """
    nframes = 0
    arks = kio.ArkReader(file_name)
    for ut, feat in arks.next_ark():
        nframes += feat.shape[0]
    sec = nframes * shift_sz * 1e-3  # i.e. shift_sz in msec
    return sec


def utterance_mean_mean(lang_path, attrib_type):
    """
    Return mean of attributes in language cluster:
    mean across utterance and across language cluster
    """
    cnt_ut = 0
    lang_mean = np.zeros((len(ATTRIBUTES_CLS[attrib_type])))
    for f in scan_folder(lang_path, attrib_type):
        arks = kio.ArkReader(f)
        for ut, feat in arks.next_ark():
            bin = metr.step(feat, 0.5)  # TODO try without binarization
            m = np.mean(bin, axis=0)
            lang_mean += m
            cnt_ut += 1
    # average across all files
    # calculation: tot_mean /= cnt_arks
    lang_mean = np.exp(np.log(lang_mean) - np.log(cnt_ut))
    return lang_mean


def utterance_global_mean(lang_path, attrib_type, threshold=False):
    """
    Return mean of attributes in language cluster:
    mean across utterance and across language cluster
    """
    cnt_frame = 0
    accum = np.zeros((len(ATTRIBUTES_CLS[attrib_type])))
    for f in scan_folder(lang_path, attrib_type):
        arks = kio.ArkReader(f)
        for ut, feat in arks.next_ark():
            if threshold:
                feat = metr.step(feat, 0.5)
            m = np.sum(feat, axis=0)
            accum += m
            cnt_frame += feat.shape[0]
    # average across all files
    # calculation: tot_mean /= cnt_arks
    lang_mean = np.exp(np.log(accum) - np.log(cnt_frame))
    return lang_mean


def utterance_histogram(lang_path, attrib_type, bins=N_HIST):
    dmin = 0.
    dmax = 1.
    ncls = len(ATTRIBUTES_CLS[attrib_type])
    bins_arr = np.linspace(dmin, dmax, bins)
    accum_hist = np.zeros((ncls, bins - 1))

    for f in scan_folder(lang_path, attrib_type):
        arks = kio.ArkReader(f)
        for ut, feat in arks.next_ark():
            harr = []
            for at_id in xrange(ncls):
                htmp, bin_edges = np.histogram(feat[:, at_id], bins_arr)
                harr.append(htmp)

            accum_hist += np.array(harr)
    return accum_hist


def language_utt_num_async(lang_paths, attrib_type, n_jobs, backend=None):
    res = {}
    with jl.Parallel(n_jobs=n_jobs, verbose=2, backend=backend) as parallel:
        for lp in lang_paths:
            cnt = parallel(jl.delayed(utterance_number)(f)
                           for f in scan_folder(lp, attrib_type))
            res[path.split(lp)[-1]] = sum(cnt)
    return res


def language_utt_len_async(lang_paths, attrib_type, n_jobs, backend=None):
    res = {}
    with jl.Parallel(n_jobs=n_jobs, verbose=2, backend=backend) as parallel:
        for lp in lang_paths:
            ts = parallel(jl.delayed(utterance_length)(f)
                          for f in scan_folder(lp, attrib_type))
            res[path.split(lp)[-1]] = sum(ts) / 3600.
    return res


def language_mean_mean_async(lang_paths, attrib_type, n_jobs, backend=None):
    res = {}
    ts = jl.Parallel(n_jobs=n_jobs, verbose=2, backend=backend)(
        jl.delayed(utterance_mean_mean)(lp, attrib_type) for lp in lang_paths)
    for m, lang in zip(ts, lang_paths):
        res[path.split(lang)[-1]] = m
    return res


def language_global_mean_async(lang_paths, attrib_type, n_jobs, backend=None):
    res = {}
    ts = jl.Parallel(n_jobs=n_jobs, verbose=2, backend=backend)(
        jl.delayed(utterance_global_mean)(lp, attrib_type) for lp in lang_paths)
    for m, lang in zip(ts, lang_paths):
        res[path.split(lang)[-1]] = m
    return res

def language_hist_async(lang_paths, attrib_type, n_jobs, backend=None):
    res = {}
    ts = jl.Parallel(n_jobs=n_jobs, verbose=2, backend=backend)(
        jl.delayed(utterance_histogram)(lp, attrib_type) for lp in lang_paths)
    for m, lang in zip(ts, lang_paths):
        res[path.split(lang)[-1]] = m
    return res


def update_store(store, langs, stat, name):
    for ln in langs:
        store[ln][name] = stat[ln]


def check_key(ddict, key):
    return not ddict or (key not in ddict[lang_dirs[0]])


def prepare_mean(stats_store, langs, stat, scale=True):
    lang_mean = []
    for ld in langs:
        lang_mean.append(stats_store[ld][stat])
    lang_mean = np.array(lang_mean)
    mean_clean = np.delete(lang_mean, cls_id_del, axis=1)
    if scale:
        accum = np.sum(mean_clean, axis=1)
        mean_clean /= accum[:, np.newaxis]
    return mean_clean


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    # ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    # ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Attributes')


if __name__ == '__main__':
    attrib_type = 'place'
    dump_file = '%s_stat.pkl' % attrib_type
    cls_filter = ['other', 'silence']
    cls_clean = [a for a in ATTRIBUTES_CLS[attrib_type] if a not in cls_filter]
    cls_id_del = [ATTRIBUTES_CLS[attrib_type].index(f) for f in cls_filter]

    debug = False

    if debug:
        n_jobs = 2
        root_path = './data/lre'
        lang_dirs = np.sort(os.listdir(root_path)).tolist()
    else:
        n_jobs = 20
        root_path = '/sipudata/pums/ivan/projects_data/mulan_lre17/train'
        lang_dirs = np.sort(['ara-acm', 'ara-ary', 'eng-gbr', 'por-brz', 'qsl-rus',
                             'spa-eur', 'zho-cmn', 'ara-apc', 'ara-arz', 'eng-usg', 'qsl-pol',
                             'spa-car', 'spa-lac', 'zho-nan'])

    # if dump exist, just update it
    if os.path.isfile(dump_file):
        pkl_file = open(dump_file, 'rb')
        stats_store = pickle.load(pkl_file)
        pkl_file.close()
    else:
        stats_store = collections.defaultdict(dict)

    lang_paths = [path.join(root_path, ld) for ld in lang_dirs]
    # ===
    # number of utterance per language
    # ===
    # if empty, update storage;
    # if not empty and no key, update
    if check_key(stats_store, Stats.num):
        start = time.time()
        r = language_utt_num_async(lang_paths, attrib_type, n_jobs, 'multiprocessing')
        end = time.time()
        print('Calc number || : %f' % (end - start))
        update_store(stats_store, lang_dirs, r, Stats.num)
    # ===
    # hours per each language
    # ===
    if check_key(stats_store, Stats.len):
        start = time.time()
        r = language_utt_len_async(lang_paths, attrib_type, n_jobs, 'multiprocessing')
        end = time.time()
        print('Calc length || : %f' % (end - start))
        update_store(stats_store, lang_dirs, r, Stats.len)
    # ===
    # mean of mean across utterances
    # ===
    if check_key(stats_store, Stats.mean_mean):
        start = time.time()
        r = language_mean_mean_async(lang_paths, attrib_type, len(lang_dirs), 'multiprocessing')
        end = time.time()
        print('Calc mean_mean || : %f' % (end - start))
        update_store(stats_store, lang_dirs, r, Stats.mean_mean)
        print(dict(stats_store))
    # ===
    # global mean across utterances
    # ===
    if check_key(stats_store, Stats.gmean):
        start = time.time()
        r = language_global_mean_async(lang_paths, attrib_type, len(lang_dirs), 'multiprocessing')
        end = time.time()
        print('Calc global_mean || : %f' % (end - start))
        update_store(stats_store, lang_dirs, r, Stats.gmean)
        print(dict(stats_store))
    # ===
    # continues mean across utterances
    # ===
    # if check_key(stats_store, Stats.cont_mean):
    #     start = time.time()
    #     r = language_global_mean_async(lang_paths, attrib_type, len(lang_dirs), 'multiprocessing')
    #     end = time.time()
    #     print('Calc %s || : %f' % (Stats.cont_mean, (end - start)))
    #     update_store(stats_store, lang_dirs, r, Stats.cont_mean)
    #     print(dict(stats_store))
    # ===
    # Histograms
    # ===
    if check_key(stats_store, Stats.hist):
        start = time.time()
        r = language_hist_async(lang_paths, attrib_type, len(lang_dirs), 'multiprocessing')
        end = time.time()
        print('Calc %s || : %f' % (Stats.hist, (end - start)))
        update_store(stats_store, lang_dirs, r, Stats.hist)

    # ===
    # dump statistics
    # ===
    out_file = open(dump_file, 'wb')
    pickle.dump(stats_store, out_file)
    out_file.close()

    # TODO plot violin
    # ===
    # plot statistic figures
    # ===
    # mean of mean
    # mean_clean = prepare_mean(stats_store, lang_dirs, Stats.mean_mean)
    # pltr.plot_stackbars(mean_clean, cls_clean, lang_dirs, '%s_%s.png' % (attrib_type, Stats.mean_mean))

    # global mean
    # mean_clean = prepare_mean(stats_store, lang_dirs, Stats.gmean)
    # pltr.plot_stackbars(mean_clean, cls_clean, lang_dirs, '%s_%s.png' % (attrib_type, Stats.gmean))

    # TODO finish histogram kernel approximation plots
    # TODO replace histogram with H, edges = np.histogramdd(r, bins = (5, 8, 4))

    # histogram plots
    # see ref: http://danielhnyk.cz/fitting-distribution-histogram-using-python/
    from scipy.stats.kde import gaussian_kde
    import bokeh.io as bio
    import bokeh.layouts as bl
    from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter
    from bokeh.plotting import figure
    import colorcet as cc

    def joy(category, data, scale=1.):
        return list(zip([category] * len(data), scale * data))

    bio.output_file("distribution.html")

    cats = ATTRIBUTES_CLS[attrib_type]
    palette = [cc.rainbow[i * 15] for i in range(len(cats))]
    bins = np.linspace(0., 1., N_HIST)
    delta = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    xr = np.linspace(-0.5, 1.5, 100)

    plots = []
    for ld in lang_dirs:
        p = figure(x_range=(0., 1.), y_range=cats, plot_width=400, plot_height=500, toolbar_location=None, title='Language: %s' % ld)
        nframes = stats_store[ld][Stats.len] * 3600. / 10e-3
        h = stats_store[ld][Stats.hist] / nframes
        source = ColumnDataSource(data=dict(x=xr))
        for at_id, at in enumerate(ATTRIBUTES_CLS[attrib_type]):
            pdf = gaussian_kde(h[at_id, :])
            y = joy(at, pdf(xr), scale=0.2)
            source.add(y, at)
            p.patch('x', at, color=palette[at_id], alpha=0.6, line_color="black", source=source)
        p.y_range.range_padding = 0.12
        plots.append(p)

    bio.show(bl.column(*plots))

    # ===
    # Violin Figures
    # ===
    fig, axs = plt.subplots(len(lang_dirs), 1, figsize=(20, 30), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.01, wspace=.5)
    axs = axs.ravel()
    xp = np.linspace(0., 1.1, 100)

    for i, ld in enumerate(lang_dirs):
        nframes = stats_store[ld][Stats.len] * 3600. / 10e-3
        h = stats_store[ld][Stats.hist] / nframes
        pos = range(1, 2 * len(ATTRIBUTES_CLS[attrib_type]), 2)
        yp = []
        for at_id, at in enumerate(ATTRIBUTES_CLS[attrib_type]):
            # pdf = gaussian_kde(h[at_id, :])
            # yp.append(pdf(xr))
            yp.append(h[at_id, :])

        vpart = axs[i].violinplot(yp, pos, points=20, widths=1.5, showmeans=True, showextrema=False, showmedians=False) # showmeans=True, showextrema=True, showmedians=True
        axs[i].set_ylabel(ld, rotation=0, fontsize=12, ha='right', va='center')
        axs[i].set_yticks([], [])
        axs[i].set_xticks([], [])

        for vp, c in zip(vpart['bodies'], palette):
            vp.set_facecolor(c)
            vp.set_edgecolor('black')
            vp.set_alpha(1)

    set_axis_style(axs[i], ATTRIBUTES_CLS[attrib_type])
    plt.show()
    plt.savefig('violin_%s.png' % attrib_type, bbox_inches="tight")
