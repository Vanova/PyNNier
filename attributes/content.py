"""
Speech attributes content analysis in NIST LRE 2017 dataset
"""
import os
import time
import pickle
import os.path as path
import numpy as np
import metrics.metrics as metr
import utils.kaldi.io as kio
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
import collections

plt.switch_backend('agg')
plt.style.use('seaborn')

MANNER_CLS = []
PLACE_CLS = []
ATTRIBUTES_CLS = {'manner': ['fricative', 'glides', 'nasal', 'other', 'silence', 'stop', 'voiced', 'vowel'],
                  'place': ['coronal', 'dental', 'glottal', 'high', 'labial',
                            'low', 'mid', 'other', 'palatal', 'silence', 'velar'],
                  'fusion_manner': [],
                  'fusion_place': []}


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


def language_utt_num_async(lang_paths, attrib_type, n_jobs, backend=None):
    res = {}
    with jl.Parallel(n_jobs=n_jobs, backend=backend) as parallel:
        for lp in lang_paths:
            cnt = parallel(jl.delayed(utterance_number)(f)
                           for f in scan_folder(lp, attrib_type))
            res[path.split(lp)[-1]] = sum(cnt)
    return res


def language_utt_len_async(lang_paths, attrib_type, n_jobs, backend=None):
    res = {}
    with jl.Parallel(n_jobs=n_jobs, backend=backend) as parallel:
        for lp in lang_paths:
            ts = parallel(jl.delayed(utterance_length)(f)
                          for f in scan_folder(lp, attrib_type))
            res[path.split(lp)[-1]] = sum(ts) / 3600.
    return res


def language_mean_mean_async(lang_paths, attrib_type, n_jobs, backend=None):
    res = {}
    with jl.Parallel(n_jobs=n_jobs, verbose=2, backend=backend) as parallel:
        ts = parallel(jl.delayed(utterance_mean_mean)(lp, attrib_type)
                      for lp in lang_paths)
    for m, lang in zip(ts, lang_paths):
        res[path.split(lang)[-1]] = m
    return res


def update_store(store, langs, stat, name):
    for ln in langs:
        store[ln][name] = stat[ln]


if __name__ == '__main__':
    attrib_type = 'manner'
    dump_file = '%s_stat.pkl' % attrib_type
    filter_cls = ['other', 'silence']
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
    start = time.time()
    r = language_utt_num_async(lang_paths, attrib_type, n_jobs, 'multiprocessing')
    end = time.time()
    print('Calc number || : %f' % (end - start))
    update_store(stats_store, lang_dirs, r, 'number')
    # ===
    # hours per each language
    # ===
    start = time.time()
    r = language_utt_len_async(lang_paths, attrib_type, n_jobs, 'multiprocessing')
    end = time.time()
    print('Calc length || : %f' % (end - start))
    update_store(stats_store, lang_dirs, r, 'length')
    # ===
    # distribution per each language and plotting
    # ===
    start = time.time()
    r = language_mean_mean_async(lang_paths, attrib_type, n_jobs, 'multiprocessing')
    end = time.time()
    print('Calc mean_mean || : %f' % (end - start))
    update_store(stats_store, lang_dirs, r, 'mean_mean')
    print(dict(stats_store))
    # ===
    # dump statistics
    # ===
    out_file = open(dump_file, 'wb')
    pickle.dump(stats_store, out_file)
    out_file.close()
    # pkl_file = open('%s_stat.pkl' % attrib_type, 'rb')
    # stats_store = pickle.load(pkl_file)
    # pkl_file.close()

    # TODO plot stat
    # # filter attribute classes
    # lang_mean = np.array(lang_mean)
    # id_del = [ATTRIBUTES_CLS[attrib_type].index(f) for f in filter_cls]
    # mean_clean = np.delete(lang_mean, id_del, 1)
    # cls_clean = [a for a in ATTRIBUTES_CLS[attrib_type] if a not in filter_cls]
    # # plot total mean per each language
    # df = pd.DataFrame(mean_clean, columns=cls_clean)
    # df.plot(kind='barh', stacked=True)
    # plt.yticks(range(len(lang_dirs)), lang_dirs)
    # plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    # plt.show()
    # plt.savefig(attrib_type + '.png', bbox_inches="tight")
