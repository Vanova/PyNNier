import itertools
import json
import socket
import sys
from io import open as iopen
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
import os
import os.path as path
import requests

DATA_DIR = './data/'

def read_tasking(fname):
    """
    read in tasking for files not downloaded yet
    """
    task_list = list()
    tasking = open(fname, 'r')
    for task in tasking:
        task = task.strip()
        task = json.loads(task)
        myf = path.join(DATA_DIR, task['filename'])
        if not path.isfile(myf):
            # check
            check_dir(path.dirname(myf))
            task_list.append(task)
    tasking.close()
    return task_list


def grouper(n, iterable):
    """
    put return n itmes at a time from an interable
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def work_func(task):
    """
    Download an image from a url
    return success/failure and the task to caller
    """
    try:
        # experiment with time
        # to big give laggy results
        socket.setdefaulttimeout(1)
        i = requests.get(task['url'])
    except:
        return((False, task))

    if i.status_code == requests.codes.ok:
        filename = path.join(DATA_DIR, '{0}'.format(task['filename']))
        f = iopen(filename, 'wb')
        f.write(i.content)
        f.close()
        return((True, task))
    else:
        return((False, task))


def check_dir(dir):
    if not path.exists(dir):
        os.makedirs(dir)


def main(at_once=100):
    # file detailing what to download
    # each line should be of format:
    # fileType, color, brand, model, url, hash
    fname = sys.argv[1]

    # multiprocessing.Pool 33sec per 100 photos
    # multiprocessing.dummy.Pool ncpu*3 22sec per 100 photos
    # 500 for 120sec, 3*ncpu

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)
    NUM_WORKERS = multiprocessing.cpu_count() * 3
    p = Pool(NUM_WORKERS)
    # get listing of files to download
    task_list = read_tasking(fname)
    print('processed {0} lines to download'.format(len(task_list)))
    print('tasking loaded')

    # download in batches
    good = open(fname + '.good', 'w')
    bad = open(fname + '.bad', 'w')
    for batch in tqdm(grouper(at_once, task_list)):
        retval = p.map(work_func, batch)
        for r, t in retval:
            if r:
                good.write(json.dumps(t)+'\n')
            else:
                bad.write(json.dumps(t)+'\n')
        good.flush()
        bad.flush()

    good.close()
    bad.close()


if __name__ == '__main__':
    main()
