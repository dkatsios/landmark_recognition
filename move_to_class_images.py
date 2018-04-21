# coding: utf-8

import pickle
# from cv2 import imread, resize
# from PIL import Image
from pathlib import Path
import random
import calendar
import time
import os

from shutil import copyfile

from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
import os

working_folder = r'E:\Data_Files\Workspaces\PyCharm\kaggle\landmark-recognition-challenge\\'
train_images_folder = working_folder + r'train_images\\'
val_images_folder = working_folder + r'val_images\\'
test_images_folder = working_folder + r'test_images\\'
new_images_folder = working_folder + r'new_train_images/'
# new_images_folder = working_folder + r'new_toy_images/'

data_folder = working_folder + r'data\\'
csv_csv_path = working_folder + 'train.csv'


def save_obj(obj, name, folder=data_folder):
    with open(folder + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, folder=data_folder):
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)

csv_name_id_tuples_list = load_obj('csv_name_id_tuples_list')
csv_ids_list = load_obj('csv_ids_list')
csv_ids_set = load_obj('csv_ids_set')
csv_names_set = load_obj('csv_names_set')

csv_id_name_dict = load_obj('csv_id_name_dict')
csv_name_id_dict = load_obj('csv_name_id_dict')


train_names_list = load_obj('train_names_list')
train_name_id_dict = load_obj('train_name_id_dict')

val_names_list = load_obj('val_names_list')
val_name_id_dict = load_obj('val_name_id_dict')

train_class_images_path = working_folder + 'train_class_images//'
val_class_images_path = working_folder + 'val_class_images//'


def change_size(imgs_folder, target_size=(224, 224, 3)):
    size = (target_size[0], target_size[1])
    total_imgs = len(list(os.listdir(imgs_folder)))
    counter = 0
    for filename in os.listdir(imgs_folder):
        try:
            img = Image.open(imgs_folder + filename)
        except OSError:
            print(filename)
            continue
        img = img.resize(size, Image.ANTIALIAS)
        img.save(imgs_folder + filename)
        if counter % 100 == 0:
            print(counter, ' out of ', total_imgs)
        counter += 1


def make_class_images(folder_path, classes_path, suffix=r'.jpg'):
    for filename in os.listdir(folder_path):
        idd = csv_name_id_dict[filename.replace(suffix, '')]
        new_dir = classes_path + str(idd)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        copyfile(folder_path + filename, new_dir + r'//' + filename)


def make_class_images_ratio(folder_path, category_1_path, classes_1_path, category_2_path, classes_2_path, ratio=0.5, suffix=r'.jpg'):
    total_imgs = len(list(os.listdir(folder_path)))
    classes_1_num = int(total_imgs * ratio)
    index = 0
    for filename in os.listdir(folder_path):
        (category_path, classes_path) = (category_1_path, classes_1_path) if index < classes_1_num else (category_2_path, classes_2_path)

        idd = csv_name_id_dict[filename.replace(suffix, '')]
        new_dir = classes_path + str(idd)
        # if not os.path.exists(new_dir):
        #     os.makedirs(new_dir)
        copyfile(folder_path + filename, category_path + filename)
        copyfile(folder_path + filename, new_dir + r'/' + filename)
        # print(filename, idd)
        if index % 100 == 0:
            print(index, ' out of ', total_imgs)
        index += 1


# change_size(new_images_folder)
# make_class_images(val_images_folder, val_class_images_path)
# make_class_images(train_images_folder, train_class_images_path)
make_class_images_ratio(new_images_folder, train_images_folder, train_class_images_path,
                        val_images_folder, val_class_images_path, ratio=0.5)

print('done')
