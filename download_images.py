# -*- coding: utf-8 -*-

# !/usr/bin/python

# Note: requires the tqdm package (pip install tqdm)

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

# Thanks to @maxwell: https://www.kaggle.com/maxwell110/python3-version-image-downloader

import sys, os, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO
import tqdm


def parse_data(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


def download_image(key_url):
    out_dir = r'E:\Data_Files\Workspaces\PyCharm\kaggle\landmark-recognition-challenge\new_train_images'
    check_1_dir = r'E:\Data_Files\Workspaces\PyCharm\kaggle\landmark-recognition-challenge\train_images'
    check_2_dir = r'E:\Data_Files\Workspaces\PyCharm\kaggle\landmark-recognition-challenge\val_images'
    (key, url) = key_url
    check_1_name = os.path.join(check_1_dir, '{}.jpg'.format(key))
    check_2_name = os.path.join(check_2_dir, '{}.jpg'.format(key))
    filename = os.path.join(out_dir, '{}.jpg'.format(key))

    if os.path.exists(check_1_name) or os.path.exists(filename) or os.path.exists(check_2_name):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1
    
    return 0


def loader():

    data_file = r'E:\Data_Files\Workspaces\PyCharm\kaggle\landmark-recognition-challenge\train.csv'
    out_dir = r'E:\Data_Files\Workspaces\PyCharm\kaggle\landmark-recognition-challenge\new_train_images'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = parse_data(data_file)
    pool = multiprocessing.Pool(processes=20)  # Num of CPUs
    failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, key_url_list), total=len(key_url_list)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()


# arg1 : data_file.csv
# arg2 : output_dir
if __name__ == '__main__':
    loader()
