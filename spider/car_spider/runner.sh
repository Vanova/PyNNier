#!/bin/bash

source activate ai


#python scrape_list.py
#python download_imgs.py ./data/lists/dataset
img_root="./data"
data_list="./data/lists/dataset"
python ./verify_image_type.py $data_list  $img_root