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


if __name__ == '__main__':
    attrib_type = 'manner'
    filter_cls = ['other', 'silence']
    debug = True

    if debug:
        root_path = './data/lre'
        n_jobs = 2
    else:
        root_path = '/sipudata/pums/ivan/projects_data/mulan_lre17/train'
        n_jobs = 20

    # loop through language clusters folder and calculate stats per language
    lang_dirs = np.sort(os.listdir(root_path)).tolist()
    lang_paths = [path.join(root_path, ld) for ld in lang_dirs]
    # ===
    # number of utterance per language
    # ===
    start = time.time()
    for lp in lang_paths:
        cnt_arks = 0
        for f in scan_folder(lp, attrib_type):
            cnt_arks += utterance_number(f)
        print('Utterances: %d' % cnt_arks)
    end = time.time()
    print('Time single: %f' % (end - start))

    start = time.time()
    r = language_utt_num_async(lang_paths, attrib_type, n_jobs, 'multiprocessing')
    end = time.time()
    print('Time || : %f' % (end - start))
    print(r)

    # ===
    # hours per each language
    # ===
    start = time.time()
    for lp in lang_paths:
        tot_time = 0
        for f in scan_folder(lp, attrib_type):
            tot_time += utterance_length(f)
        print('Total length: %s' % str(tot_time / 3600.))
    end = time.time()
    print('Time single: %f' % (end - start))

    start = time.time()
    r = language_utt_len_async(lang_paths, attrib_type, n_jobs, 'multiprocessing')
    end = time.time()
    print('Time || : %f' % (end - start))
    print(r)

    # ===
    # distribution per each language and plotting
    # ===
    store_mean = {}
    # for lp in lang_paths:
    #     cnt_arks = 0
    #     lang_mean = np.zeros((len(ATTRIBUTES_CLS[attrib_type])))
    #     for f in scan_folder(lp, attrib_type):
    #         arks = kio.ArkReader(f)
    #         for ut, feat in arks.next_ark():
    #             bin = metr.step(feat, 0.5)  # TODO try without binarization
    #             m = np.mean(bin, axis=0)
    #             lang_mean += m
    #             cnt_arks += 1
    #     # average across all files
    #     # calculation: tot_mean /= cnt_arks
    #     lang_mean = np.exp(np.log(lang_mean) - np.log(cnt_arks))
    #     store_mean[path.split(lp)[-1]] = lang_mean
    # print(store_mean)
    #
    # store_mean = {}
    # for lp in lang_paths:
    #     store_mean[path.split(lp)[-1]] = {}
    #     store_mean[path.split(lp)[-1]]['mean_mean'] = utterance_mean_mean(lp, attrib_type)
    # print(store_mean)
    #
    # output = open('%s_stat.pkl' % attrib_type, 'wb')
    # pickle.dump(store_mean, output)
    # output.close()
    # pkl_file = open('%s_stat.pkl' % attrib_type, 'rb')
    # data1 = pickle.load(pkl_file)
    # pkl_file.close()
    #     # TODO dump means
    #
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
