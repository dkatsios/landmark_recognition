import pickle
from PIL import Image
import os

import numpy as np
from keras import backend as K
from keras.models import model_from_json
from keras.optimizers import Adam
import tensorflow as tf

working_folder = r'E:\Data_Files\Workspaces\PyCharm\kaggle\landmark-recognition-challenge\\'
data_folder = working_folder + r'data/'
# model_path = data_folder + '/continue_second_phase_logs/1521020813_xcpetion_model.h5'

this_model = data_folder + 'xcpetion_model_.json'
model_weights = data_folder + 'continue_second_phase_logs/older/1521093898_xcpetion_model.h5'

test_folder = working_folder + r'train_images/'
test_csv_path = working_folder + 'test.csv'
results_file = data_folder + 'train_results.csv'

batch_size = 64
input_shape = (224, 224, 3)


def load_obj(name, folder=data_folder):
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)

val_name_id_dict = load_obj('val_name_id_dict')
class_indices_dict = load_obj('class_indices_dict')
inverted_class_indices_dict = dict((v, k) for k, v in class_indices_dict.items())


def make_ids_list(folder_path, suffix='.jpg'):
    names_list = []
    for filename in os.listdir(folder_path):
        filename = filename.replace(suffix, '')
        names_list.append(filename)
    names_list.sort()
    return names_list


def make_csv_list(csv_path):
    names_list = []
    with open(csv_path) as f:
        f.readline()
        for line in f:
            l = line.replace('"', '').strip().split(',')
            if len(l) != 2:
                print(l)
                continue
            name = l[0]
            names_list.append(name)
    return names_list


def add_unknown_imgs(results_file):
    test_csv_names_list = make_csv_list(test_csv_path)
    csv_names_set = set(test_csv_names_list)
    with open(results_file, 'r') as f:
            f.readline()
            for line in f:
                l = line.strip().split(',')
                if len(l) != 2:
                    print(l)
                    continue
                name = l[0]
                csv_names_set.remove(name)

    with open(results_file, 'a') as f:
            for name in csv_names_set:
                line = '\n' + name + ','
                f.write(line)


def pred_generator(ids_list, imgs_dir, batch_size, img_size=(224, 224, 3), suffix='.jpg'):
    while len(ids_list) > 0:
        batch_size = batch_size if len(ids_list) >= batch_size else len(ids_list)
        imgs = np.zeros((batch_size, *img_size))
        for i in range(batch_size):
            name = ids_list.pop(0)
            img = Image.open(imgs_dir + name + suffix)
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


def eval_generator(ids_list, labels, labels_num, imgs_dir, batch_size, img_size=(224, 224, 3), suffix='.jpg'):
    while len(ids_list) > 0:
        batch_size = batch_size if len(ids_list) >= batch_size else len(ids_list)
        imgs = np.zeros((batch_size, *img_size))
        lbls = np.zeros((batch_size, labels_num))
        for i in range(batch_size):
            name = ids_list.pop(0)
            label_name = labels.pop(0)
            label_ind = class_indices_dict[label_name]
            img = Image.open(imgs_dir + name + suffix)
            img = np.asarray(img.resize(img_size[:2], Image.ANTIALIAS))
            img = img / 255.0

            # mean
            img_mean = np.mean(img, axis=(0, 1))
            img = img - img_mean

            #std
            img_std = np.std(img, axis=(0, 1))
            img = img / img_std
            imgs[i] = img
            lbls[i, label_ind] = 1
        yield imgs, lbls


def split_seq(seq, size):
    """ Split up seq in pieces of size """
    return [seq[i:i+size] for i in range(0, len(seq), size)]


def evaluate_results(results_file, test_folder, labels_num, batch_size, input_shape):
    eval_list = list()
    labels = list()
    with open(results_file, 'r') as f:
            f.readline()
            for line in f:
                tokens = line.strip().split(',')
                name = tokens[0]
                label = tokens[1].strip().split(' ')[0]
                eval_list.append(name)
                labels.append(label)
    eval_gen = eval_generator(eval_list, labels, labels_num, test_folder, batch_size, input_shape)
    losses = xcpetion_model.evaluate_generator(eval_gen, steps=steps)
    return losses


# class My_Dropout(Dropout):
#     def set_new_rate(self, new_rate):
#         self.rate = new_rate
#
#     def dropped_inputs(self, rate, inputs):
#         noise_shape = self._get_noise_shape(inputs)
#         return K.dropout(inputs, rate, noise_shape,
#                          seed=self.seed)
#
#     def call(self, inputs, training=None):
#         if 0. < self.rate < 1.:
#             print('other real drop ', self.rate)
#             return K.in_train_phase(self.dropped_inputs(rate=self.rate, inputs=inputs), inputs,
#                                     training=training)
#         print('real drop ', self.rate)
#         return inputs


def evaluate_val_results(results_file, test_folder, val_name_id_dict, labels_num, batch_size, input_shape):
    eval_list = list()
    labels = list()
    with open(results_file, 'r') as f:
            f.readline()
            for line in f:
                tokens = line.strip().split(',')
                name = tokens[0]
                # label = tokens[1].split(' ')[0]
                label = str(val_name_id_dict[name])
                eval_list.append(name)
                labels.append(label)
    eval_gen = eval_generator(eval_list, labels, labels_num, test_folder, batch_size, input_shape)
    losses = xcpetion_model.evaluate_generator(eval_gen, steps=steps)
    return losses


def load_model_from_file(path):
    # load json and create model
    with open(path, 'r') as json_file:
        model = model_from_json(json_file.read())
    return model


def gap(y_true, y_pred):
    arg_true = y_true
    arg_pred = y_pred
    val_pred = K.max(arg_pred, axis=-1)
    _, sorted_indices = tf.nn.top_k(val_pred, batch_size)

    new_arg_pred = []
    new_arg_true = []

    for i in range(batch_size):
        _pred = arg_pred[sorted_indices[i]]
        _true = arg_true[sorted_indices[i]]
        new_arg_pred.append(_pred)
        new_arg_true.append(_true)

    new_arg_pred = K.stack(new_arg_pred)
    new_arg_true = K.stack(new_arg_true)

    arg_pred = new_arg_pred
    arg_true = new_arg_true

    arg_pred = K.argmax(arg_pred, axis=-1)
    arg_true = K.argmax(arg_true, axis=-1)

    correct_pred = K.equal(arg_pred, arg_true)
    precision = [K.switch(K.gather(correct_pred, i - 1),
                          K.sum(K.cast(correct_pred, 'float32')[:i]) / i,
                          K.variable(0))
                 for i in range(1, batch_size + 1)]
    precision = K.stack(precision)
    _gap = K.sum(precision) / K.sum(K.cast(correct_pred, 'float32'))
    return _gap


def square_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

if __name__ == "__main__":
    test = False
    train_lim = 2**15
    pieces = 20
    whole_ids_list = make_ids_list(test_folder)
    if train_lim:
        whole_ids_list = whole_ids_list[:train_lim]
    total_ids = len(whole_ids_list)
    # xcpetion_model = load_model(model_path)

    xcpetion_model = load_model_from_file(this_model)
    xcpetion_model.load_weights(model_weights, by_name=True)
    xcpetion_model.compile(optimizer=Adam(lr=0.0002), loss=square_error, metrics=['acc', gap])
    print(xcpetion_model.summary())
    total = len(split_seq(whole_ids_list, int(total_ids / pieces)))
    with open(results_file, 'w') as f:
        f.write('id,landmarks')

    for counter, ids_list in enumerate(split_seq(whole_ids_list, int(total_ids / pieces))):
        steps = int(len(ids_list) / batch_size)
        steps += 0 if len(ids_list) % batch_size == 0 else 1

        pred_list = ids_list[:]
        pred_gen = pred_generator(pred_list, test_folder, batch_size, input_shape)
        predicts = xcpetion_model.predict_generator(pred_gen, steps=steps, verbose=1)
        certainties = np.max(predicts, axis=-1)
        labels_inds = np.argmax(predicts, axis=-1)

        with open(results_file, 'a') as f:
            text = ''
            for i in range(len(ids_list)):
                text += "\n" + str(ids_list[i]) + "," + inverted_class_indices_dict[labels_inds[i]] + " " +\
                        str(certainties[i])
            f.write(text)

        print('done {} out of {}'.format(counter + 1, total))

    if test:
        add_unknown_imgs(results_file)
    # else:
    #     labels_num = xcpetion_model.layers[-1].output_shape[1]
        # losses = evaluate_results(results_file, test_folder, labels_num, batch_size, input_shape)

        # losses = evaluate_val_results(results_file, test_folder, val_name_id_dict,
        #                               labels_num, batch_size, input_shape)
        # print(xcpetion_model.metrics_names)
        # print(losses)








