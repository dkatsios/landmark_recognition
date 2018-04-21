# coding: utf-8

import pickle
from cv2 import imread, resize
from PIL import Image
from numbers import Number
import os
from  os import listdir
from pathlib import Path
import multiprocessing
from scipy.spatial.distance import cosine, euclidean, hamming

import numpy as np
from keras.layers import Input, Conv2D, Dropout, merge, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras import layers
from keras.models import Model, Sequential
from keras.models import load_model, model_from_json
import keras
from keras import backend as K
import tensorflow as tf
import pandas as pd

working_folder = r'E:\Data_Files\Workspaces\PyCharm\kaggle\landmark-recognition-challenge/'
train_images_folder = working_folder + r'train_images/'
val_images_folder = working_folder + r'val_images/'
test_images_folder = working_folder + r'test_images/'
toy_val_folder = working_folder + r'toy_val_images/'

data_folder = working_folder + r'data/'
train_codes_folder = data_folder + 'codes/'
test_codes_folder = data_folder + 'test_codes_folder/'
codes_by_category_folder = data_folder + 'codes_by_category/'
siamese_first_phase_folder = data_folder + r'siamese_first_phase_folder/'
siamese_second_phase_folder = data_folder + r'siamese_second_phase_folder/'
siamese_third_phase_folder = data_folder + r'siamese_third_phase_folder/'

siamese_model_weights = siamese_first_phase_folder + '1521927456_siamese_model.h5'

csv_csv_path = working_folder + 'train.csv'

train_class_images_path = working_folder + 'train_class_images/'
val_class_images_path = working_folder + 'val_class_images/'

siamese_weights_path = siamese_first_phase_folder + '1521927456_siamese_model.h5'
results_file = data_folder + 'best_test_results.csv'
new_results_file = data_folder + 'sim_results_.csv'

val_codes_folder = data_folder + 'val_codes/'
val_results_file = data_folder + 'val_results.csv'
new_val_results_file = data_folder + 'sim_val_results_.csv'


def print_func(func):
    def echo_func(*func_args, **func_kwargs):
        print('')
        print('Excecuting func: {}'.format(func.__name__))
        return func(*func_args, **func_kwargs)
    return echo_func


def bg(gen):
    def _bg_gen(gen, conn):
        while conn.recv():
            try:
                conn.send(next(gen))
            except StopIteration:
                conn.send(StopIteration)
                return

    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_bg_gen, args=(gen, child_conn))
    p.start()

    parent_conn.send(True)
    while True:
        parent_conn.send(True)
        x = parent_conn.recv()
        if x is StopIteration:
            return
        else:
            yield x


def save_obj(obj, name, folder=data_folder):
    with open(folder + name.replace('.pkl', '') + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, folder=data_folder):
    with open(folder + name.replace('.pkl', '') + '.pkl', 'rb') as f:
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
input_shape = img_size
code_size = 1024
suf = '.jpg'


def make_folder_lists_dicts(folder_path, suffix=r'.jpg'):
    names_list = []
    name_id_dict = dict()
    id_name_dict = dict()
    for filename in listdir(folder_path):
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


def make_codes_by_category(train_codes_folder, codes_by_category_folder, train_name_id_dict):
    counter = 1
    for train_codes in listdir(train_codes_folder):
        print(train_codes)
        counter += 1
        temp_cats_dict = dict()
        train_codes = load_obj(train_codes, folder=train_codes_folder)
        for name, code in zip(train_codes['names'], train_codes['codes']):
            cat = train_name_id_dict[name.replace('.jpg', '')]
            if cat not in temp_cats_dict.keys():
                temp_cats_dict[cat] = dict()
                temp_cats_dict[cat]['names'] = []
                temp_cats_dict[cat]['codes'] = []
            temp_cats_dict[cat]['names'].append(name)
            temp_cats_dict[cat]['codes'].append(code)
        for cat in temp_cats_dict.keys():
            if not Path(codes_by_category_folder + str(cat)).is_file():
                cat_file = dict()
                cat_file['names'] = []
                cat_file['codes'] = []
                save_obj(obj=cat_file, name=str(cat), folder=codes_by_category_folder)
            cat_file = load_obj(str(cat), folder=codes_by_category_folder)
            names = temp_cats_dict[cat]['names']
            codes = temp_cats_dict[cat]['codes']
            cat_file['names'].extend(names)
            cat_file['codes'].extend(codes)
            save_obj(obj=cat_file, name=str(cat), folder=codes_by_category_folder)


# @bg
def get_codes_from_files(codes_folder):
    """ yields the codes (dict) from each file in the codes folder"""
    for filename in listdir(codes_folder):
        codes = load_obj(filename, folder=codes_folder)
        yield codes


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


def compute_codes(imgs_path, codes_folder, siamese_weights_path):
    batch_size = 64
    code_net = get_code_net(siamese_weights_path)

    codes_batches_size = 2**14
    print('make images names list')
    imgs_names = listdir(imgs_path)
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
            save_obj(code_dict, 'batch_{}'.format(counter), folder=codes_folder)
            this_names = []
            print('done {} out of {}'.format(counter, batches_num))
            counter += 1
    print('codes computed and saved at: ' + codes_folder)


def make_categories_vectors(codes_folder, name_id_dict, imgs_per_cat_limit=None):
    """returns a dict with keys the categories (ids) of the images and values the 1024 size vectors for each category"""
    suf = '.jpg'
    cat_codes_dict = dict()
    for codes in get_codes_from_files(codes_folder):
        for ind, name in enumerate(codes['names']):
            idd = name_id_dict[name.replace(suf, '')]
            if idd not in cat_codes_dict.keys():
                cat_codes_dict[idd] = [np.zeros(code_size), 0]
            if imgs_per_cat_limit and cat_codes_dict[idd][1] >= imgs_per_cat_limit:
                continue
            cat_codes_dict[idd][0] += codes['codes'][ind]
            cat_codes_dict[idd][1] += 1
    for key, value in cat_codes_dict.items():
        value[0] /= value[1]
        cat_codes_dict[key] = value[0]
    return cat_codes_dict


@print_func
def make_categories_vectors_2(codes_folder, name_id_dict, imgs_per_cat_limit=None):
    """returns a dict with keys the categories (ids) of the images and values the 1024 size vectors for each category"""
    suf = '.jpg'
    cat_codes_dict = dict()
    counter = 1
    for codes in get_codes_from_files(codes_folder):
        for ind, name in enumerate(codes['names']):
            idd = name_id_dict[name.replace(suf, '')]
            if idd not in cat_codes_dict.keys():
                cat_codes_dict[idd] = [np.zeros(code_size), 0]
            if imgs_per_cat_limit and cat_codes_dict[idd][1] >= imgs_per_cat_limit:
                continue
            code = codes['codes'][ind] >= .5
            code = code.astype(float)
            cat_codes_dict[idd][0] += code
            cat_codes_dict[idd][1] += 1
        print('code_file', counter)
        counter += 1
    for key, value in cat_codes_dict.items():
        val = value[0] / value[1]
        val = val >= .5
        val = val.astype(float)
        cat_codes_dict[key] = val
    return cat_codes_dict


def get_results(results_file):
    """returns a dict with images names (without suffix) as keys and the result's idd/class as value"""
    names_results_ids_dict = dict()
    with open(results_file) as f:
        f.readline()
        for line in f:
            name, idd = line.strip().split(',')
            if len(idd) < 3:
                save_obj(names_results_ids_dict, 'names_results_ids_dict')
                return names_results_ids_dict
            idd = idd.split(' ')[0]
            names_results_ids_dict[name] = idd
    return names_results_ids_dict


def get_code_net(siamese_weights_path):
    siamese_net = SiameseModel(data_folder + '2nd_phase_xcpetion_model.h5')
    siamese_net.load_weights(siamese_weights_path)
    short_model = siamese_net.get_layer('model_1')

    xcpetion_model = load_model_from_file(data_folder + 'xcpetion_model_.json')
    xcpetion_model.get_layer('dense_1').activation = K.sigmoid
    code_net = Model(inputs=xcpetion_model.input, outputs=xcpetion_model.get_layer('dense_1').output)

    code_net = load_weights_from_model(code_net, short_model)
    return code_net


def load_model_from_file(path):
    # load json and create model
    with open(path, 'r') as json_file:
        model = model_from_json(json_file.read())
    return model


def load_weights_from_model(model, old_model):
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Dropout, keras.layers.InputLayer)):
            continue
        print(layer.name)
        weights = old_model.get_layer(layer.name).get_weights()
        model.get_layer(layer.name).set_weights(weights)
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


class GetBatchCodes:
    def __init__(self, siamese_model_weights, imgs_folder, img_size):
        siamese_net = SiameseModel(data_folder + '2nd_phase_xcpetion_model.h5')
        siamese_net.load_weights(siamese_model_weights)
        short_model = siamese_net.get_layer('model_1')

        xcpetion_model = load_model_from_file(data_folder + 'xcpetion_model_.json')
        self.code_net = Model(inputs=xcpetion_model.input, outputs=xcpetion_model.get_layer('dense_1').output)

        self.code_net = load_weights_from_model(self.code_net, short_model)
        self.imgs_folder = imgs_folder
        self.suf = '.jpg'
        self.img_size = img_size

    def __call__(self, img_names):
        img_names = [name.replace(self.suf, '') for name in img_names]
        imgs = np.zeros((len(img_names), *self.img_size))
        for ind, img_name in enumerate(img_names):
            img = Image.open(self.imgs_folder + img_name + self.suf)
            img = np.asarray(img.resize(self.img_size[:2], Image.ANTIALIAS))
            img = img / 255.0

            # mean
            img_mean = np.mean(img, axis=(0, 1))
            img = img - img_mean

            # std
            img_std = np.std(img, axis=(0, 1))
            img = img / img_std

            imgs[ind] = img
        codes = self.code_net.predict(imgs)
        return codes


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


def get_res_codes(cats, cat_codes_dict, code_size):
    """returns a numpy array of size (categories, code size) with the general codes for the given categories"""
    cat_codes = np.zeros((len(cats), code_size))
    for ind, cat in enumerate(cats):
        cat_codes[ind] = cat_codes_dict[int(cat)]
    return cat_codes


def chunks(l1, l2, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l1), n):
        yield (l1[i:i + n], l2[i:i + n])


def cos_similarity(name_codes, ids_codes):
    """takes two numpy arrays and returns a list of size equal to the rows of the arrays
    with the cos similarity between the rows fo the two arrays"""
    cos_sims = []
    for i in range(name_codes.shape[0]):
        code_1 = name_codes[i]
        code_2 = ids_codes[i]
        cos_sim = cosine(code_1, code_2)
        cos_sims.append(cos_sim)
    return cos_sims


def write_new_results(names, ids, similarities, new_results_file):
    assert len(names) == len(similarities),\
        'problem with sizes!\nnames size {} different than similarities size {}'.format(len(names), len(similarities))
    with open(new_results_file, 'w') as f:
        f.write('id,landmarks')
        text = ''
        for i in range(len(names)):
            text += "\n" + str(names[i]) + "," + str(ids[i]) + " " + str(similarities[i])
        f.write(text)


def change_results(cat_codes_dict, imgs_folder, results_file, new_results_file):
    print('start getting results codes')
    names_results_ids_dict = get_results(results_file)
    get_batch_codes = GetBatchCodes(siamese_model_weights, imgs_folder, img_size)
    batch_size = 64
    img_similarities = []
    imgs_codes = []
    keys = list(names_results_ids_dict.keys())
    values = list(names_results_ids_dict.values())
    counter = 1
    total = len(keys) // batch_size
    for names, res_ids in chunks(keys, values, batch_size):
        name_codes = get_batch_codes(names)
        ids_codes = get_res_codes(res_ids, cat_codes_dict, code_size)
        img_similarities.extend(cos_similarity(name_codes, ids_codes))
        imgs_codes.extend(name_codes)
        print('done {} of {}'.format(counter, total))
        counter += 1
    save_obj(imgs_codes, 'test_images_codes')
    save_obj(img_similarities, 'img_similarities')
    write_new_results(keys, values, img_similarities, new_results_file)


def calc_new_prob(old_prob, sim_prob, old_prop_rate=0.5):
    old_prob, sim_prob = float(old_prob), float(sim_prob)
    # if old_prob > .9:
    #     return old_prob
    # if sim_prob > .9:
    #     return sim_prob
    new_prob = (old_prob * old_prop_rate) + (sim_prob * (1 - old_prop_rate))
    return new_prob


@print_func
def change_results_from_code_files(test_img_similarity_dict, results_file, new_results_file, old_prop_rate=0.5):
    with open(results_file) as fin:
        with open(new_results_file, 'w') as fout:
            fout.write(fin.readline())
            for line in fin:
                name, idd = line.strip().split(',')
                if len(idd.split(' ')) == 2:
                    old_prob = idd.split(' ')[1]
                    sim_prob = str(test_img_similarity_dict[name])
                    new_prob = calc_new_prob(old_prob, sim_prob, old_prop_rate)
                    line = line.replace(old_prob, str(new_prob))
                fout.write(line)


def distance_to_similarity(test_img_similarity_dict, min_distance):
    for key, value in test_img_similarity_dict.items():
        test_img_similarity_dict[key] = min_distance / value
    return test_img_similarity_dict


# @print_func
def compute_similarities(test_codes_folder, cat_codes_dict, results_file):
    """works best for now with ratio 0.9"""
    test_img_similarity_dict = dict()
    names_results_ids_dict = get_results(results_file)
    counter = 1
    for codes_dict in get_codes_from_files(test_codes_folder):
        names = codes_dict['names']
        codes = codes_dict['codes']
        for name, code in zip(names, codes):
            name = name.replace(suf, '')
            cat = names_results_ids_dict[name]
            cat_code = cat_codes_dict[int(cat)]
            similarity = 1 - cosine(code, cat_code)
            test_img_similarity_dict[name] = similarity
            # min_distance = distance if distance < min_distance else min_distance
        print(counter)
        counter += 1
    # test_img_similarity_dict = distance_to_similarity(test_img_similarity_dict, min_distance)
    save_obj(test_img_similarity_dict, 'test_img_similarity_dict')
    return test_img_similarity_dict


def calc_for_each_sim(code, cat):
    cat_codes = load_obj(cat, folder=codes_by_category_folder)
    total = len(cat_codes['codes'])
    similarity = 0
    for cat_code in cat_codes['codes']:
        similarity += 1 - cosine(code, cat_code)
    return similarity / total


# @print_func
def compute_similarities_for_each(test_codes_folder, cat_codes_dict, results_file):
    test_img_similarity_dict = dict()
    names_results_ids_dict = get_results(results_file)
    counter = 1
    for codes_dict in get_codes_from_files(test_codes_folder):
        for name, code in zip(codes_dict['names'], codes_dict['codes']):
            name = name.replace(suf, '')
            cat = names_results_ids_dict[name]
            similarity = calc_for_each_sim(code, cat)
            test_img_similarity_dict[name] = similarity
            # min_distance = distance if distance < min_distance else min_distance
        print(counter)
        counter += 1
    # test_img_similarity_dict = distance_to_similarity(test_img_similarity_dict, min_distance)
    save_obj(test_img_similarity_dict, 'test_img_similarity_dict')
    return test_img_similarity_dict


@print_func
def compute_similarities_2(test_codes_folder, cat_codes_dict, results_file):
    """Tried cosine with boolean->float and hamming with boolean.
    Simple cosine with the plain cat_code and old ration of .9 is best for now"""
    test_img_similarity_dict = dict()
    names_results_ids_dict = get_results(results_file)
    counter = 1
    for codes_dict in get_codes_from_files(test_codes_folder):
        names = codes_dict['names']
        codes = codes_dict['codes']
        for name, code in zip(names, codes):
            name = name.replace(suf, '')
            cat = names_results_ids_dict[name]
            cat_code = cat_codes_dict[int(cat)]
            code, cat_code = code >= .5, cat_code >= .5
            # code, cat_code = code.astype(float), cat_code.astype(float)
            similarity = 1 - hamming(code, cat_code)
            test_img_similarity_dict[name] = similarity
            # min_distance = distance if distance < min_distance else min_distance
        print(counter)
        counter += 1
    # test_img_similarity_dict = distance_to_similarity(test_img_similarity_dict, min_distance)
    save_obj(test_img_similarity_dict, 'test_img_similarity_dict')
    return test_img_similarity_dict


def extract_from_results_file(val_results_file):
    names, cats, probs = [], [], []
    with open(val_results_file) as f:
        f.readline()
        for line in f:
            name, idd = line.strip().split(',')
            if len(idd.split(' ')) != 2:
                continue
            cat, prob = idd.split(' ')
            names.append(name)
            cats.append(cat)
            probs.append(prob)
    return names, cats, probs


def get_correct_cats(names, cats, print_it=False):
    bool_correct_cats = []
    for ind, name in enumerate(names):
        correct_cat = val_name_id_dict[name]
        bool_correct_cats.append(int(cats[ind]) == correct_cat)
    if print_it:
        print('correct categories: {}%'.format(sum(bool_correct_cats) * 100 / len(names)))
    return bool_correct_cats


def gap_from_bools(bools):
    gap = 0
    corrects = 0
    for ind, elem in enumerate(bools):
        if elem:
            corrects += 1
            gap += corrects / (ind + 1)
    return gap / len(bools)


def calc_gap(bool_correct_cats, probs):
    zipped = zip(bool_correct_cats, probs)
    zipped = sorted(zipped, key=lambda x: x[1])
    sorted_bool_correct_cats, sorted_probs = zip(*zipped)
    gap = gap_from_bools(sorted_bool_correct_cats)
    return gap


def compute_gap(val_results_file):
    names, cats, probs = extract_from_results_file(val_results_file)
    bool_correct_cats = get_correct_cats(names, cats)
    gap = calc_gap(bool_correct_cats, probs)
    return gap


@print_func
def val_old_new_rate_test(original_gap):
    cat_codes_dict = load_obj('cat_codes_dict', folder=data_folder)
    img_similarity_dict = compute_similarities_for_each(val_codes_folder, cat_codes_dict, val_results_file)
    # old_prop_rate = .8
    for i in range(11):
        old_prop_rate = i / 10.
        change_results_from_code_files(img_similarity_dict, val_results_file, new_val_results_file, old_prop_rate)
        gap = compute_gap(new_val_results_file)
        print(old_prop_rate, gap)
        print('yes' if gap > original_gap else 'no')


@print_func
def run_val_tests():
    load_files()
    gap = compute_gap(val_results_file)
    print(gap)
    val_old_new_rate_test(gap)


@print_func
def run_trains():
    load_files()
    gap = compute_gap(val_results_file)
    print(gap)
    val_old_new_rate_test(gap)


def run_tests():
    load_files()
    cat_codes_dict = load_obj('cat_codes_dict', folder=data_folder)
    img_similarity_dict = compute_similarities_for_each(test_codes_folder, cat_codes_dict, results_file)
    old_prop_rate = 0
    change_results_from_code_files(img_similarity_dict, results_file, new_results_file, old_prop_rate)


if __name__ == '__main__':
    print('start excecution')
    # run_val_tests()
    run_tests()
    run_trains()
    # make_codes_by_category(train_codes_folder, codes_by_category_folder, train_name_id_dict)

    # print('start excecution')
    # load_files()

    # cat_codes_dict = make_categories_vectors_2(val_codes_folder, train_name_id_dict)
    # save_obj(cat_codes_dict, name='cat_codes_dict_2', folder=data_folder)

    # cat_codes_dict = load_obj('cat_codes_dict', folder=data_folder)
    # compute_codes(val_images_folder, val_codes_folder, siamese_weights_path)

    # gap = compute_gap(val_results_file)
    # print(gap)

    # change_results(cat_codes_dict, test_images_folder, results_file, new_results_file)
    # test_img_similarity_dict = compute_similarities_2(test_codes_folder, cat_codes_dict, results_file)

    # cat_codes_dict = load_obj('cat_codes_dict', folder=data_folder)
    # test_img_similarity_dict = compute_similarities(val_codes_folder, cat_codes_dict, val_results_file)
    # test_img_similarity_dict = load_obj('test_img_similarity_dict', folder=data_folder)
    # old_prop_rate = .8
    # change_results_from_code_files(test_img_similarity_dict, results_file, new_results_file, old_prop_rate)


