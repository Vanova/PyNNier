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


def scan_folder(lang_dir, attrib_cls):
    print('Language dir: %s' % lang_dir)
    flist = []
    for j in xrange(n_jobs):
        p = '%s/res/%s/scores.%i.txt' % (lang_dir, attrib_cls, j + 1)
        flist.append(p)
    return flist


def score_mean(filter_cls, binarise=True):
    pass


# def scores_content():
# + number of files per each language

# - hours per each language

# distributions of scores th = 0.5

# score_file = path.join(ROOT_PATH, p)
# score = kio.read_ark_file(score_file) # TODO make iterator


# define sampling: because some languages have more speech data
# compare others

if __name__ == '__main__':
    root_path = '/sipudata/pums/ivan/projects_data/mulan_lre17/train'
    type_at = 'place'
    filter_cls = ['other', 'silence']
    # window 25ms, shift 10ms
    wnd = 0.025
    shift = 2
    n_jobs = 20
    debug = False

    if debug:
        file_name = '../utils/kaldi/manner.ark'
        arks = kio.ArkReader(file_name)
        cnt_arks = 0
        for ut, feat in arks.next_ark():
            cnt_arks += 1
            print('File: %s' % ut)
            print(feat.shape)
        print('Total: %d' % cnt_arks)

        # fast calculate number of utterances
        cnt_arks = 0
        match = kio.ArkReader.grep(file_name, '[')
        cnt_arks += len(match)
        print('Total: %d' % cnt_arks)

        # length in hours
        arks = kio.ArkReader(file_name)
        nframes = 0
        for ut, feat in arks.next_ark():
            nframes += feat.shape[0]
        sec = nframes * wnd / shift
        print('Total length: %s' % str(sec / 3600.))

        # distribution: mean of file and across files
        # TODO think about sampling
        arks = kio.ArkReader(file_name)
        all_mean = np.zeros((1, 8))
        cnt_arks = 0
        for ut, feat in arks.next_ark():
            bin = metr.step(feat, 0.5)
            m = np.mean(bin, axis=0, keepdims=True)
            all_mean += m
            cnt_arks += 1
        # save calculation: tot_mean /= cnt_arks
        all_mean = np.exp(np.log(all_mean) - np.log(cnt_arks))
        print(all_mean)

        # filter attribute classes
        id_del = [ATTRIBUTES_CLS['manner'].index(f) for f in filter_cls]
        all_mean = np.delete(all_mean, id_del, 1)
        cls_clean = [a for a in ATTRIBUTES_CLS['manner'] if a not in filter_cls]

        # plot total mean per each language
        df = pd.DataFrame(np.array([all_mean.squeeze(), all_mean.squeeze()]), columns=cls_clean)
        df.plot(kind='barh', stacked=True)
        plt.yticks(range(2), ['lang1', 'lang2'])
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.show()
        plt.savefig('manner.png', bbox_inches="tight")

    else:
        # loop through language clusters folder and calculate stats per language
        lang_dirs = np.sort(os.listdir(root_path))
        lang_mean = []
        # filter attribute classes
        id_del = [ATTRIBUTES_CLS[type_at].index(f) for f in filter_cls]
        for ldir in lang_dirs:
            # ===
            # number of utterance per language
            # ===
            # cnt_arks = 0
            # for f in scan_folder(ldir, 'manner'):
            #     match = kio.ArkReader.grep(path.join(root_path, f), '[')
            #     cnt_arks += len(match)
            # print('Utterances: %d' % cnt_arks)
            # # ===
            # # hours per each language
            # # ===
            # nframes = 0
            # for f in scan_folder(ldir, 'manner'):
            #     arks = kio.ArkReader(path.join(root_path, f))
            #     for ut, feat in arks.next_ark():
            #         nframes += feat.shape[0]
            # sec = nframes * wnd / shift
            # print('Total length: %s' % str(sec / 3600.))
            # ===
            # distribution per each language and plotting
            # ===
        # TODO filter other and silence
            cnt_arks = 0
            all_mean = np.zeros((1, len(ATTRIBUTES_CLS[type_at])))
            for f in scan_folder(ldir, type_at):
                arks = kio.ArkReader(path.join(root_path, f))
                for ut, feat in arks.next_ark():
                    bin = metr.step(feat, 0.5)  # TODO without binarization
                    m = np.mean(bin, axis=0, keepdims=True)
                    all_mean += m
                    cnt_arks += 1
            # average across all files
            # save calculation: tot_mean /= cnt_arks
            all_mean = np.exp(np.log(all_mean) - np.log(cnt_arks))
            lang_mean.append(all_mean.squeeze())
            print(all_mean)
        # filter attribute classes
        lang_mean = np.array(lang_mean)
        mean_clean = np.delete(lang_mean, id_del, 1)
        cls_clean = [a for a in ATTRIBUTES_CLS[type_at] if a not in filter_cls]
        # plot total mean per each language
        df = pd.DataFrame(mean_clean, columns=cls_clean)
        df.plot(kind='barh', stacked=True)
        plt.yticks(range(len(lang_dirs)), lang_dirs)
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.savefig(type_at + '.png', bbox_inches="tight")
