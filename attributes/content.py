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
    # window 25ms, shift 10ms
    wnd = 0.025
    shift = 2
    n_jobs = 20
    debug = True

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
        sec = nframes * wnd/shift
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

        # plot total mean per each language
        df = pd.DataFrame(all_mean, columns=ATTRIBUTES_CLS['manner'])
        df.plot(kind='barh', stacked=True)
        plt.show()
        plt.savefig('manner.png')
    else:
        # loop through language clusters folder and calculate stats per language
        lang_dirs = np.sort(os.listdir(root_path))
        for ldir in lang_dirs:
            # ===
            # number of utterance per language
            # ===
            cnt_arks = 0
            for f in scan_folder(ldir, 'manner'):
                match = kio.ArkReader.grep(path.join(root_path, f), '[')
                cnt_arks += len(match)
            print('Utterances: %d' % cnt_arks)
            # ===
            # hours per each language
            # ===
            nframes = 0
            for f in scan_folder(ldir, 'manner'):
                arks = kio.ArkReader(path.join(root_path, f))
                for ut, feat in arks.next_ark():
                    nframes += feat.shape[0]
            sec = nframes * wnd / shift
            print('Total length: %s' % str(sec / 3600.))
            # ===
            # distribution per each language and plotting
            # ===

