"""
Speech attributes content analysis in NIST LRE 2017 dataset
"""
import os
import os.path as path
import numpy as np
import utils.kaldi.io as kio

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
    # number of files per each language

    # hours per each language

    # distributions of scores th = 0.5

    # score_file = path.join(ROOT_PATH, p)
    # score = kio.read_ark_file(score_file) # TODO make iterator


# define sampling: because some languages have more speech data
# compare others

if __name__ == '__main__':
    root_path = '/sipudata/pums/ivan/projects_data/mulan_lre17/train'
    n_jobs = 20
    debug = False

    if debug:
        file_name = '../utils/kaldi/test.ark'
        ark_iter = kio.ArkReader(file_name)
        for id, arr in ark_iter.next_ark():
            print('File: %s' % id)
            print(arr.shape)
    else:
        # loop through language clusters folder and calculate stats per language
        lang_dirs = np.sort(os.listdir(root_path))
        for ldir in lang_dirs:
            flist = scan_folder(ldir, 'manner')

            for f in flist:
                ark_iter = kio.ArkReader(path.join(root_path, f))
                for id, arr in ark_iter.next_ark():
                    print('File: %s' % id)
                    print(arr.shape)
