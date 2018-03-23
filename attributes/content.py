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

MANNER_CLS = []
PLACE_CLS = []
ATTRIBUTES_CLS = {'manner': ['fricative', 'glides', 'nasal', 'other', 'silence', 'stop', 'voiced', 'vowel'],
                  'place': ['coronal', 'dental', 'glottal', 'high', 'labial',
                            'low', 'mid', 'other', 'palatal', 'silence', 'velar'],
                  'fusion_manner': [],
                  'fusion_place': []}
N_HIST = 20


class Stats(object):
    num = 'number'
    len = 'length'
    mean_mean = 'mean_mean'
    gmean = 'global_mean'
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


def utterance_global_mean(lang_path, attrib_type):
    """
    Return mean of attributes in language cluster:
    mean across utterance and across language cluster
    """
    cnt_frame = 0
    accum = np.zeros((len(ATTRIBUTES_CLS[attrib_type])))
    for f in scan_folder(lang_path, attrib_type):
        arks = kio.ArkReader(f)
        for ut, feat in arks.next_ark():
            bin = metr.step(feat, 0.5)  # TODO try without binarization
            m = np.sum(bin, axis=0)
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


if __name__ == '__main__':
    attrib_type = 'manner'
    dump_file = '%s_stat.pkl' % attrib_type
    cls_filter = ['other', 'silence']
    cls_clean = [a for a in ATTRIBUTES_CLS[attrib_type] if a not in cls_filter]
    cls_id_del = [ATTRIBUTES_CLS[attrib_type].index(f) for f in cls_filter]

    debug = True

    if debug:
        root_path = './data/lre'
        n_jobs = 2
    else:
        root_path = '/sipudata/pums/ivan/projects_data/mulan_lre17/train'
        n_jobs = 20
    # if dump exist, just update it
    if os.path.isfile(dump_file):
        pkl_file = open(dump_file, 'rb')
        stats_store = pickle.load(pkl_file)
        pkl_file.close()
    else:
        stats_store = collections.defaultdict(dict)

    lang_dirs = np.sort(os.listdir(root_path)).tolist()
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
    # Histograms
    # ===
    if not check_key(stats_store, Stats.hist):
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
    mean_clean = prepare_mean(stats_store, lang_dirs, Stats.mean_mean)
    pltr.plot_stackbars(mean_clean, cls_clean, lang_dirs, '%s_%s.png' % (attrib_type, Stats.mean_mean))

    # global mean
    mean_clean = prepare_mean(stats_store, lang_dirs, Stats.gmean)
    pltr.plot_stackbars(mean_clean, cls_clean, lang_dirs, '%s_%s.png' % (attrib_type, Stats.gmean))

    # histogram
    # bins_arr = np.linspace(0., 1., N_HIST)
    # width = 0.7 * (bins_arr[1] - bins_arr[0])
    # center = (bins_arr[:-1] + bins_arr[1:]) / 2
    # for ld in ['lan1']:
    #     h = stats_store[ld][Stats.hist]
    #     for at_id in xrange(len(ATTRIBUTES_CLS[attrib_type])):
    #         # plt.hist(harr[:, at_id], 100, alpha=0.5)
    #         # dt = h[at_id, :] /
    #         plt.bar(center, h[at_id, :], align='center', width=width, alpha=0.5,
    #                 label=ATTRIBUTES_CLS[attrib_type][at_id])
    #
    # plt.legend(loc='upper right')
    # plt.show()


