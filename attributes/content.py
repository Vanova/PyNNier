"""
Speech attributes content analysis in NIST LRE 2017 dataset
"""
import os
import os.path as path
import numpy as np
import metrics.metrics as metr
import utils.kaldi.io as kio
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('agg')
plt.style.use('seaborn')

MANNER_CLS = []
PLACE_CLS = []
ATTRIBUTES_CLS = {'manner': ['fricative', 'glides', 'nasal', 'other', 'silence', 'stop', 'voiced', 'vowel'],
                  'place': ['coronal', 'dental', 'glottal', 'high', 'labial',
                            'low', 'mid', 'other', 'palatal', 'silence', 'velar'],
                  'fusion_manner': [],
                  'fusion_place': []}


def scan_folder(lang_dir, attrib_cls):  # TODO scan by name tamplate!!!
    print('Language dir: %s' % lang_dir)
    flist = []
    for j in xrange(n_jobs):
        p = '%s/res/%s/scores.%i.txt' % (lang_dir, attrib_cls, j + 1)
        flist.append(p)
    return flist


def async_task():
    pass


# @timeit
def job_score_mean(filter_cls, binarise=True):
    pass


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
    lang_dirs = np.sort(os.listdir(root_path))
    # ===
    # number of utterance per language
    # ===
    for ldir in lang_dirs:
        cnt_arks = 0
        for f in scan_folder(ldir, attrib_type):
            cnt_arks += utterance_number(path.join(root_path, f))
        print('Utterances: %d' % cnt_arks)

    # ===
    # hours per each language
    # ===
    for ldir in lang_dirs:
        tot_time = 0
        for f in scan_folder(ldir, attrib_type):
            tot_time += utterance_length(path.join(root_path, f))
        print('Total length: %s' % str(tot_time / 3600.))
    # ===
    # distribution per each language and plotting
    # ===
    lang_mean = []
    for ldir in lang_dirs:
        cnt_arks = 0
        all_mean = np.zeros((1, len(ATTRIBUTES_CLS[attrib_type])))
        for f in scan_folder(ldir, attrib_type):
            arks = kio.ArkReader(path.join(root_path, f))
            for ut, feat in arks.next_ark():
                bin = metr.step(feat, 0.5)  # TODO without binarization
                m = np.mean(bin, axis=0, keepdims=True)
                all_mean += m
                cnt_arks += 1
        # average across all files
        # calculation: tot_mean /= cnt_arks
        all_mean = np.exp(np.log(all_mean) - np.log(cnt_arks))
        lang_mean.append(all_mean.squeeze())
        print(all_mean)
        # TODO dump means

    # filter attribute classes
    lang_mean = np.array(lang_mean)
    id_del = [ATTRIBUTES_CLS[attrib_type].index(f) for f in filter_cls]
    mean_clean = np.delete(lang_mean, id_del, 1)
    cls_clean = [a for a in ATTRIBUTES_CLS[attrib_type] if a not in filter_cls]
    # plot total mean per each language
    df = pd.DataFrame(mean_clean, columns=cls_clean)
    df.plot(kind='barh', stacked=True)
    plt.yticks(range(len(lang_dirs)), lang_dirs)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.show()
    plt.savefig(attrib_type + '.png', bbox_inches="tight")
