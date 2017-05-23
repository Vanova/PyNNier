import json
from hashlib import md5
from urllib2 import Request
from urllib2 import urlopen
from bs4 import BeautifulSoup
from tqdm import tqdm
import config as cfg


def get_car(color, brand, model, num, out_file, out_error):
    """
    For a specific make model and color of car attempt to get 'num' images
    """
    make_p = brand.replace(' ', '+')
    model_p = model.replace(' ', '+')
    query_str = '{0}+{1}+{2}'.format(color, make_p, model_p)

    # query with no usage image rights
    # url = 'https://www.google.co.in/search?q=' + query + '&source=lnms&tbm=isch'

    # return images with appropriate usage rights
    url = 'https://www.google.co.in/search?q=' + query_str + '&tbs=sur:fc&tbm=isch'
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 ' +
                      '(KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36'}
    try:
        # TODO check color of the cars and num
        soup = get_soup(url, header)
        actual_images = []
        # parse all photo urls
        for a in soup.find_all('div', {'class': 'rg_meta'}):
            link, link_type = json.loads(a.text)['ou'], json.loads(a.text)['ity']
            actual_images.append((link, link_type))

        for i, (link, link_type) in enumerate(actual_images[:num]):
            if link_type is not None:
                # write out where to get the image from
                data = format_line(color, brand, model, link, link_type)
                out_file.write(json.dumps(data) + '\n')
    except:
        # spout some error messages when things go poorly
        data = format_line(color, brand, model, link, 'FAIL')
        out_error.write(json.puts(data) + '\n')


def get_soup(url, header):
    """
    Download a file with a given user-agent string
    """
    return BeautifulSoup(urlopen(Request(url, headers=header)), 'html.parser')


def format_line(color, make, model, img, file_type):
    """
    Format item metadata
    """
    d = dict()
    d['brand'] = make
    d['model'] = model
    d['color'] = color
    text = '' + color + make + model + img
    m = md5()
    m.update(text.encode('utf-8'))
    d['hash'] = m.hexdigest()
    fname = '{0}/{1}/{2}.{3}'.format(color, make, d['hash'], file_type)
    d['filename'] = fname.replace(' ', '_')
    d['url'] = img
    return d


def read_cars(filename):
    """
    Read the brand and model from a file.
    File format: brand, model
    """
    cars = dict()
    bm = open(filename, 'r')
    allbm = bm.readlines()
    bm.close()
    for item in allbm:
        br, ml = item.split(',', 1)
        brand = br.strip()
        model = ml.strip()
        if brand not in cars:
            cars[brand] = set()
        cars[brand].add(model)
    return cars


def read_colors(filename):
    """
    Read the acceptable colors from a file.
    File format: color
    """
    c = open(filename, 'r')
    allc = c.readlines()
    c.close()
    colors = [i.strip() for i in allc]
    return colors


def main(images_per_model=100):
    cars = read_cars(cfg.CAR_LIST)
    colors = read_colors(cfg.COLOR_LIST)

    out_file = open(cfg.DATASET_LIST, 'w')
    out_error = open(cfg.ERRORS_LIST, 'w')

    # all permutations of cars and colors
    for color in tqdm(colors):
        for brand in cars.keys():
            for model in cars[brand]:
                # time.sleep(2)
                get_car(color, brand, model, images_per_model, out_file, out_error)
                out_file.flush()
                out_error.flush()

    out_file.close()
    out_error.close()


if __name__ == '__main__':
    main()
