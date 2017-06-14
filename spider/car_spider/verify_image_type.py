import functools
import json
import sys
from multiprocessing import Pool
from keras.preprocessing import image
import cv2


def readTasking(filename):
    """
    """
    tasking = open(filename, 'r')
    data = list()

    for task in tasking:
        task = task.strip()
        line = json.loads(task)
        data.append(line)
    tasking.close()
    return data


def procLine2(l, r):
    """
    If the file is image, resize and save
    """
    img_path = '{0}/{1}'.format(r, l['filename'])
    try:
        small_img = image.load_img(img_path, target_size=(224, 224))
        small_img.save(img_path)
        return (True, l['filename'])
    except:
        return (False, l['filename'])


def writeTasking(filename, tasking, bad):
    """
    """
    outFile = open(filename, 'w')
    badFiles = set()
    for item in bad:
        if not item[0]:
            badFiles.add(item[1])
    for task in tasking:
        if task['filename'] not in badFiles:
            outFile.write(json.dumps(task) + '\n')

    outFile.close()


def main():
    # ===
    # take root path to image data
    # ===
    img_root = sys.argv[2]
    procLine = functools.partial(procLine2, r=img_root)

    # ===
    # read 'dataset' list and apply function 'procLine2' to it
    # ===
    data_list = sys.argv[1]
    tasking = readTasking(data_list)
    p = Pool()
    files = p.map(procLine, tasking)
    writeTasking(data_list + '.verif', tasking, files)

if __name__ == '__main__':
    main()
