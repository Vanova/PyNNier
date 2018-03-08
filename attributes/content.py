"""
Speech attributes content analysis in NIST LRE 2017 dataset
"""
import os
import os.path as path
import numpy as np
import utils.kaldi.io as kio

MANNER_CLS = []
PLACE_CLS = []
ATTRIBUTES_CLS = {'manner': [''],
              'place': ['']}
ROOT_PATH = '/sipudata/pums/ivan/projects_data/mulan_lre17/train'
n_jobs = 20


def scan_folder(attrib_cls):
    # loop lang dir
    cluster_files = {}
    lang_dirs = np.sort(os.listdir(ROOT_PATH))
    for ldir in lang_dirs:
        print('Language dir: %s' % ldir)
        flist = []
        for j in xrange(n_jobs):
            p = '%s/res/%s/scores.%i.txt' % (ldir, attrib_cls, j + 1)
            flist.append(p)
        cluster_files[ldir] = flist
    return cluster_files


def scores_content():
    # number of files
    # hours
    # distributions of scores th = 0.5
    # score_file = path.join(ROOT_PATH, p)
    # score = kio.read_ark_file(score_file) # TODO make iterator
    pass

# define sampling: because some languages have more speech data
# compare others

if __name__ == '__main__':
    # test lazy ark files reading
    file_name = '../utils/kaldi/test.ark'
    ark = kio.read_ark_file(file_name)

    ark_iter = kio.ArkReader(file_name)
    for id, arr in ark_iter.next_ark():
        print('File: %s' % id)
        print(arr.shape)
