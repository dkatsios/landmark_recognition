# coding: utf-8

import pickle
from cv2 import imread, resize
from PIL import Image
from pathlib import Path
import numpy.random as rng
from random import sample
from numbers import Number
import random
import calendar
import time
import os

import numpy as np
from keras.layers import Input, Conv2D, Dropout, merge, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras import layers
from keras.models import Model, Sequential
from keras import backend as K
from keras.models import load_model, model_from_json
from keras.optimizers import Adam
import keras
import tensorflow as tf

from similarity import get_code_net

working_folder = r'E:\Data_Files\Workspaces\PyCharm\kaggle\landmark-recognition-challenge/'
train_images_folder = working_folder + r'train_images/'
val_images_folder = working_folder + r'val_images/'
test_images_folder = working_folder + r'test_images/'
toy_val_folder = working_folder + r'toy_val_images/'

data_folder = working_folder + r'data/'
train_codes_folder = data_folder + 'codes/'
val_codes_folder = data_folder + 'val_codes/'

siamese_first_phase_folder = data_folder + r'siamese_first_phase_folder/'
siamese_second_phase_folder = data_folder + r'siamese_second_phase_folder/'
siamese_third_phase_folder = data_folder + r'siamese_third_phase_folder/'

siamese_weights_path = siamese_first_phase_folder + '1521927456_siamese_model.h5'

csv_csv_path = working_folder + 'train.csv'

train_class_images_path = working_folder + 'train_class_images/'
val_class_images_path = working_folder + 'val_class_images/'

first_phase_model_weights = siamese_first_phase_folder + '1521927456_siamese_model.h5'


def save_obj(obj, name, folder=data_folder):
    with open(folder + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, folder=data_folder):
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)


train_names_list = list()
train_name_id_dict = dict()
train_id_name_dict = dict()

val_names_list = list()
val_name_id_dict = dict()
val_id_name_dict = dict()

csv_names_list = load_obj('csv_names_list')
csv_name_id_tuples_list = load_obj('csv_name_id_tuples_list')
csv_ids_list = load_obj('csv_ids_list')
csv_ids_set = load_obj('csv_ids_set')
csv_names_set = load_obj('csv_names_set')

csv_id_name_dict = load_obj('csv_id_name_dict')
csv_name_id_dict = load_obj('csv_name_id_dict')
classes_num = len(csv_ids_set)

img_size = (224, 224, 3)


def make_folder_lists_dicts(folder_path, suffix=r'.jpg'):
    names_list = []
    name_id_dict = dict()
    id_name_dict = dict()
    for filename in os.listdir(folder_path):
        filename = filename.replace(suffix, '')
        if filename not in csv_names_set:
            continue
        names_list.append(filename)
        name_id_dict[filename] = csv_name_id_dict[filename]
        idd = name_id_dict[filename]
        if idd in id_name_dict.keys():
            id_name_dict[idd].add(filename)
        else:
            id_name_dict[idd] = {filename}
    return names_list, name_id_dict, id_name_dict


def make_csv_files():
    global csv_names_set, csv_names_list, csv_ids_set, csv_ids_list, \
        csv_id_name_dict, csv_name_id_dict, csv_name_id_tuples_list

    with open(csv_csv_path) as f:
        f.readline()
        for line in f:
            l = line.replace('"', '').strip().split(',')
            if len(l) != 3:
                print(l)
                continue
            name, idd = l[0], int(l[2])
            csv_name_id_tuples_list.append((name, idd))
            csv_names_list.append(name)

            csv_name_id_dict[name] = idd
            csv_ids_list.append(idd)

            if idd in csv_id_name_dict.keys():
                csv_id_name_dict[idd].add(name)
            else:
                csv_id_name_dict[idd] = {name}

    csv_names_set = set(csv_names_list)
    csv_ids_set = set(csv_ids_list)

    save_obj(csv_name_id_tuples_list, 'csv_name_id_tuples_list')
    save_obj(csv_names_list, 'csv_names_list')
    save_obj(csv_ids_list, 'csv_ids_list')

    save_obj(csv_name_id_dict, 'csv_name_id_dict')
    save_obj(csv_id_name_dict, 'csv_id_name_dict')

    save_obj(csv_ids_set, 'csv_ids_set')
    save_obj(csv_names_set, 'csv_names_set')


def make_files():
    global train_names_list, train_name_id_dict, train_id_name_dict,\
           val_names_list, val_name_id_dict, val_id_name_dict

    train_names_list, train_name_id_dict, train_id_name_dict = make_folder_lists_dicts(train_images_folder)
    val_names_list, val_name_id_dict, val_id_name_dict = make_folder_lists_dicts(val_images_folder)

    save_obj(train_names_list, 'train_names_list')
    save_obj(train_name_id_dict, 'train_name_id_dict')
    save_obj(train_id_name_dict, 'train_id_name_dict')

    save_obj(val_names_list, 'val_names_list')
    save_obj(val_name_id_dict, 'val_name_id_dict')
    save_obj(val_id_name_dict, 'val_id_name_dict')


def load_files():
    global train_names_list, train_name_id_dict, train_id_name_dict, \
        val_names_list, val_name_id_dict, val_id_name_dict

    train_names_list = load_obj('train_names_list')
    train_name_id_dict = load_obj('train_name_id_dict')
    train_id_name_dict = load_obj('train_id_name_dict')

    val_names_list = load_obj('val_names_list')
    val_name_id_dict = load_obj('val_name_id_dict')
    val_id_name_dict = load_obj('val_id_name_dict')

# make_files()
load_files()

batch_size = 8
input_shape = (224, 224, 3)

first_phase_epochs = 5
second_phase_epochs = 5
third_phase_epochs = 5

saves_per_epoch = 10
small_epochs = 20

imgs_per_rep = len(train_names_list) // saves_per_epoch
imgs_per_small_epoch = imgs_per_rep // small_epochs
steps_per_small_epoch = imgs_per_small_epoch // batch_size

first_phase_train_reps = first_phase_epochs * saves_per_epoch
second_phase_train_reps = second_phase_epochs * saves_per_epoch
third_phase_train_reps = third_phase_epochs * saves_per_epoch

val_size = len(val_names_list)
val_imgs_per_rep = val_size // saves_per_epoch
val_imgs_per_small_epoch = val_imgs_per_rep // small_epochs
val_steps_per_small_epoch = val_imgs_per_small_epoch // batch_size
# def W_init(shape, name=None):
#     """Initialize weights as in paper"""
#     values = rng.normal(loc=0,scale=1e-2,size=shape)
#     return K.variable(values,name=name)
#
#
# def b_init(shape, name=None):
#     """Initialize bias as in paper"""
#     values=rng.normal(loc=0.5,scale=1e-2,size=shape)
#     return K.variable(values,name=name)


def add_noise(img):
    mean = 0.5
    var = 0.05
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy = img + gauss
    img = noisy.clip(0, 1)
    return img


def siamese_generator(id_name_dict, imgs_dir, batch_size, img_size=(224, 224, 3), suffix='.jpg'):
    while True:
        imgs_0 = np.zeros((batch_size, *img_size))
        imgs_1 = np.zeros((batch_size, *img_size))

        targets = np.zeros((batch_size, 1))
        targets[:batch_size // 2, 0] = 1
        idds = sample(id_name_dict.keys(), batch_size)
        # idds = rng.choice(list(id_name_dict.keys()), size=batch_size, replace=False)

        for i in range(batch_size):
            idd = idds[i]
            same = False
            if i < batch_size // 2:
                try:
                    name_pair = sample(id_name_dict[idd], 2)
                    same = False
                except ValueError:
                    name = list(id_name_dict[idd])[0]
                    name_pair = [name, name]
                    same = True
            else:
                id_1 = idds[i]
                name_1 = sample(id_name_dict[id_1], 1)[0]

                id_2 = sample(set(id_name_dict.keys()).difference({id_1}), 1)[0]
                name_2 = sample(id_name_dict[id_2], 1)[0]
                name_pair = [name_1, name_2]

            img_pair = [Image.open(imgs_dir + name + suffix) for name in name_pair]
            img_pair = [np.asarray(img.resize(img_size[:2], Image.ANTIALIAS)) for img in img_pair]
            img_pair = [img / 255.0 for img in img_pair]

            # mean
            img_mean_pair = [np.mean(img, axis=(0, 1)) for img in img_pair]
            img_pair = [img - img_mean for img, img_mean in zip(img_pair, img_mean_pair)]

            #std
            img_std_pair = [np.std(img, axis=(0, 1)) for img in img_pair]
            img_pair = [img / img_std for img, img_std in zip(img_pair, img_std_pair)]
            imgs_0[i] = img_pair[0]
            imgs_1[i] = img_pair[1] if not same else add_noise(img_pair[1])
        yield [imgs_0, imgs_1], targets


def make_generators():
    train_img_gen = siamese_generator(train_id_name_dict, train_images_folder, batch_size, img_size=input_shape)
    val_img_gen = siamese_generator(val_id_name_dict, val_images_folder, batch_size, img_size=input_shape)
    return train_img_gen, val_img_gen


def load_model_from_file(path):
    # load json and create model
    with open(path, 'r') as json_file:
        model = model_from_json(json_file.read())
    return model


def set_trainables(model, choice):
    if isinstance(choice, Number):
        ratio = choice
        trainable_layers_index = int(len(model.layers) * (1 - ratio))
        for layer in model.layers[:trainable_layers_index]:
            layer.trainable = False
        for layer in model.layers[trainable_layers_index:]:
            layer.trainable = True
    else:
        for layer in model.layers:
            layer.trainable = layer.name in choice
    return model


def load_model_from_file(path):
    # load json and create model
    with open(path, 'r') as json_file:
        model = model_from_json(json_file.read())
    return model


def SiameseModel(base_model_path):
    xcpetion_model = load_model_from_file(data_folder + 'xcpetion_model_.json')
    xcpetion_model.load_weights(data_folder + '2nd_phase_xcpetion_weights.h5', by_name=False)
    xcpetion_model.get_layer('dense_1').activation = K.sigmoid

    short_xception_model = Model(inputs=xcpetion_model.input, outputs=xcpetion_model.get_layer('dense_1').output)
    short_xception_model = set_trainables(short_xception_model, ('dense_1'))

    print(short_xception_model.summary())

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        encoded_l = short_xception_model(left_input)
        encoded_r = short_xception_model(right_input)

    difference = layers.subtract([encoded_l, encoded_r])
    merged = layers.multiply([difference, difference])
    prediction = Dense(1, activation='sigmoid', name='prediction_dense', trainable=True)(merged)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    return siamese_net


def siamese_first_phase():
    train_img_siamese_gen, val_img_siamese_gen = make_generators()

    siamese_net = SiameseModel(data_folder + '2nd_phase_xcpetion_model.h5')
    siamese_net.load_weights(first_phase_model_weights)
    print(siamese_net.summary())

    optimizer = Adam(0.0005)
    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])

    for i in range(first_phase_train_reps):
        history = siamese_net.fit_generator(train_img_siamese_gen,
                                               steps_per_epoch=steps_per_small_epoch,
                                               epochs=small_epochs, verbose=1,
                                               validation_data=val_img_siamese_gen,
                                               validation_steps=val_steps_per_small_epoch,
                                               workers=4)
        print(i, 'out of ', first_phase_train_reps)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        siamese_net.save_weights(siamese_first_phase_folder + str(ts) + '_siamese_model.h5')
        save_obj(history.history, str(ts) + '_siamese_history.h5', folder=siamese_first_phase_folder)

    siamese_net.save(data_folder + '1st_phase_siamese_model.h5')


def siamese_second_phase():
    train_img_siamese_gen, val_img_siamese_gen = make_generators()

    siamese_net = SiameseModel(data_folder + '2nd_phase_xcpetion_model.h5')
    siamese_net.load_weights(first_phase_model_weights)
    print(siamese_net.summary())

    optimizer = Adam(0.0001)
    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])

    for i in range(first_phase_train_reps):
        history = siamese_net.fit_generator(train_img_siamese_gen,
                                               steps_per_epoch=2,
                                               epochs=2, verbose=1,
                                               validation_data=val_img_siamese_gen,
                                               validation_steps=val_steps_per_small_epoch,
                                               workers=4)
        print(i, 'out of ', first_phase_train_reps)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        siamese_net.save_weights(siamese_second_phase_folder + str(ts) + '_siamese_model.h5')
        save_obj(history.history, str(ts) + '_siamese_history.h5', folder=siamese_second_phase_folder)

    siamese_net.save(data_folder + '2nd_phase_siamese_model.h5')


def load_weights_from_file(model, old_model_file, weights_file):
    old_model = load_model_from_file(old_model_file)
    old_model.load_weights(weights_file)
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Dropout, keras.layers.InputLayer)):
            continue
        print(layer.name)
        weights = old_model.get_layer(layer.name).get_weights()
        model.get_layer(layer.name).set_weights(weights)
    return model


def load_weights_from_model(model, old_model):
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Dropout, keras.layers.InputLayer)):
            continue
        print(layer.name)
        weights = old_model.get_layer(layer.name).get_weights()
        model.get_layer(layer.name).set_weights(weights)
    return model


def batch_compute_codes(model, batch):
    batch = np.array(batch)
    codes = model.predict(batch, batch_size=batch_size)
    return codes


def img_to_array(img_path):
    img = Image.open(img_path)
    img = np.asarray(img.resize(img_size[:2], Image.ANTIALIAS))
    img = img / 255.0

    # mean
    img_mean = np.mean(img, axis=(0, 1))
    img = img - img_mean

    # std
    img_std = np.std(img, axis=(0, 1))
    img = img / img_std
    return img


def code_generator(names, imgs_dir, batch_size=64):
    while len(names) > 0:
        batch_size = batch_size if len(names) >= batch_size else len(names)
        imgs = np.zeros((batch_size, *img_size))
        for i in range(batch_size):
            name = names.pop(0)
            img = Image.open(imgs_dir + name)
            img = np.asarray(img.resize(img_size[:2], Image.ANTIALIAS))
            img = img / 255.0

            # mean
            img_mean = np.mean(img, axis=(0, 1))
            img = img - img_mean

            #std
            img_std = np.std(img, axis=(0, 1))
            img = img / img_std

            imgs[i] = img
        yield imgs
# def save_model_to_file(model, path):
#     with open(path, "w") as json_file:
#         json_file.write(model.to_json())


# def get_code_net():
#     siamese_net = SiameseModel(data_folder + '2nd_phase_xcpetion_model.h5')
#     siamese_net.load_weights(first_phase_model_weights)
#     short_model = siamese_net.get_layer('model_1')
#
#     xcpetion_model = load_model_from_file(data_folder + 'xcpetion_model_.json')
#     xcpetion_model.get_layer('dense_1').activation = K.sigmoid
#     code_net = Model(inputs=xcpetion_model.input, outputs=xcpetion_model.get_layer('dense_1').output)
#
#     code_net = load_weights_from_model(code_net, short_model)
#     return code_net


def compute_codes(imgs_path):
    batch_size = 64
    code_net = get_code_net(siamese_weights_path)

    codes_batches_size = 2**14
    print('make images names list')
    imgs_names = os.listdir(imgs_path)
    imgs_num = len(imgs_names)
    batches_num = imgs_num // codes_batches_size + (1 if imgs_num % codes_batches_size else 0)
    counter = 1
    this_names = []
    print('start loop')
    for ind, name in enumerate(imgs_names):
        this_names.append(name)
        # img = img_to_array(imgs_path + name)
        # imgs.append(img)
        if len(this_names) == codes_batches_size or ind == len(imgs_names) - 1:
            code_dict = {'names': this_names}
            names = this_names[:]
            steps = len(this_names) // batch_size + (1 if len(names) % batch_size else 0)
            codes = code_net.predict_generator(code_generator(names, imgs_path, batch_size=batch_size),
                                               steps=steps, verbose=1, workers=4)
            code_dict['codes'] = codes
            # codes = batch_compute_codes(code_net, imgs)
            save_obj(code_dict, 'batch_{}'.format(counter), folder=train_codes_folder)
            this_names = []
            print('done {} out of {}'.format(counter, batches_num))
            counter += 1


# siamese_first_phase()
compute_codes(val_images_folder)
print('this is the end')
# siamese_second_phase()

# code_dict = load_obj('batch_7', folder=codes_folder)





