import os
import os.path as path
import numpy as np
import utils.kaldi.io as kio

MANNER_CLS = []
PLACE_CLS = []

ATTRIBUTES_CLS = {'manner': [''],
              'place': ['']}

root_path = '/sipudata/pums/ivan/projects_data/mulan_lre17/train'
n_jobs = 20

# loop lang dir
lang_dirs = np.sort(os.listdir(root_path))
for ldir in lang_dirs:
    print('Language dir: %s' % ldir)
    for j in xrange(n_jobs):
        for att in ATTRIBUTES_CLS:
            p = '%s/res/%s/scores.%i.txt' % (ldir, att, j + 1)
            score_file = path.join(root_path, p)
            score = kio.read_ark_file(score_file) # TODO make iterator



# define sampling: because some languages have more speech data
# compare others
