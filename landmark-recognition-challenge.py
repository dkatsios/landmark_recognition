# coding: utf-8
import pickle
from cv2 import imread, resize
from PIL import Image
from pathlib import Path
import random
import calendar
import time
import os
from collections import Counter
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))

import numpy as np
import tensorflow as tf
from sklearn.metrics import label_ranking_average_precision_score
from keras.applications.xception import Xception
from keras.layers import Input, Conv2D, Dropout, merge, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, InputLayer
from keras.models import Model, Sequential
from keras import backend as K
from keras.models import load_model, model_from_json
from keras.optimizers import Adam
K.image_data_format() == 'channels_last'
from keras.utils import generic_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import TensorBoard, Callback
from keras.losses import categorical_hinge, mean_squared_error, mean_absolute_error, categorical_crossentropy
import keras

import losswise
from losswise.libs import LosswiseKerasCallback
losswise.set_api_key('JWN8A6X96')


working_folder = r'E:\Data_Files\Workspaces\PyCharm\kaggle\landmark-recognition-challenge\\'
train_images_folder = working_folder + r'train_images\\'
val_images_folder = working_folder + r'val_images\\'
test_images_folder = working_folder + r'test_images\\'
toy_test_images_path = working_folder + r'toy_test_images'

data_folder = working_folder + r'data\\'
first_phase_folder = data_folder + r'first_phase_logs/'
second_phase_folder = data_folder + r'second_phase_logs/'
second_second_phase_folder = data_folder + r'second_second_phase_logs/'
continue_second_phase_folder = data_folder + r'continue_second_phase_logs/'
third_phase_folder = data_folder + r'third_phase_logs/'
smoothed_third_phase_folder = data_folder + r'smoothed_third_phase_logs/'
double_drop_hinge_phase_folder = data_folder + 'double_drop_hinge_phase_logs/'
drop_ase_xcpetion_phase_folder = data_folder + 'drop_ase_phase_logs/'



this_model = data_folder + "xcpetion_model_dropout_2048_1024.json"
this_model_weights = data_folder + 'continue_second_phase_logs/older/1520948479_xcpetion_model.h5'


second_second_phase_model = second_second_phase_folder + '1520424331_xcpetion_model.h5'
double_drop_hinge_phase_model = double_drop_hinge_phase_folder + '1520855530_xcpetion_model.h5'
drop_ase_xcpetion_model_path = drop_ase_xcpetion_phase_folder + '1520589478_xcpetion_model.h5'
initial_model = smoothed_third_phase_folder + '_xcpetion_model.h5'

csv_csv_path = working_folder + 'train.csv'

train_class_images_path = working_folder + 'train_class_images//'
val_class_images_path = working_folder + 'val_class_images//'


def save_obj(obj, name, folder=data_folder):
    with open(folder + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, folder=data_folder):
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)

csv_name_id_tuples_list = []
csv_name_id_dict = dict()
csv_id_name_dict = dict()
csv_names_list = []
csv_ids_list = []

csv_name_id_tuples_list = []
csv_ids_list = []
csv_ids_set = set()
csv_names_set = set()

csv_id_name_dict = dict()
csv_name_id_dict = dict()
train_names_list = dict()
train_name_id_dict = dict()
classes_num = 0

# xcpetion_model = None
# new_xcpetion_model = None
# train_img_class_gen = None
# val_img_class_gen = None
# optimizer = None
# drop_rate = 0


def make_folder_lists_dicts(folder_path, suffix=r'.jpg'):
    names_list = []
    name_id_dict = dict()
    for filename in os.listdir(folder_path):
        filename = filename.replace(suffix, '')
        if filename not in csv_names_set:
            print(filename)
            continue
        names_list.append(filename)
        name_id_dict[filename] = csv_name_id_dict[filename]
    return names_list, name_id_dict


def make_files():
    global csv_name_id_tuples_list, csv_ids_list, csv_ids_set, csv_names_set, \
    csv_id_name_dict, csv_name_id_dict, classes_num, \
    train_names_list, train_name_id_dict, val_names_list, val_name_id_dict

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
                csv_id_name_dict[id].add(name)
            else:
                csv_id_name_dict[id] = {name}

    csv_names_set = set(csv_names_list)
    csv_ids_set = set(csv_ids_list)

    save_obj(csv_name_id_tuples_list, 'csv_name_id_tuples_list')
    save_obj(csv_names_list, 'csv_names_list')
    save_obj(csv_ids_list, 'csv_ids_list')

    save_obj(csv_name_id_dict, 'csv_name_id_dict')
    save_obj(csv_id_name_dict, 'csv_id_name_dict')

    save_obj(csv_ids_set, 'csv_ids_set')
    save_obj(csv_names_set, 'csv_names_set')

    train_names_list, train_name_id_dict = make_folder_lists_dicts(train_images_folder)
    val_names_list, val_name_id_dict = make_folder_lists_dicts(val_images_folder)

    save_obj(train_names_list, 'train_names_list')
    save_obj(train_name_id_dict, 'train_name_id_dict')

    save_obj(val_names_list, 'val_names_list')
    save_obj(val_name_id_dict, 'val_name_id_dict')


def load_files():
    global csv_name_id_tuples_list, csv_ids_list, csv_ids_set, csv_names_set, \
    csv_id_name_dict, csv_name_id_dict, classes_num, \
    train_names_list, train_name_id_dict, val_names_list, val_name_id_dict

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
    classes_num = len(csv_ids_set)

# make_files()
# print('done making files')
load_files()

batch_size = 32
input_shape = (224, 224, 3)

first_phase_epochs = 5
second_phase_epochs = 5
third_phase_epochs = 5

saves_per_epoch = 10
small_epochs = 50

imgs_per_rep = int(len(train_names_list) / saves_per_epoch)
imgs_per_small_epoch = int(imgs_per_rep / small_epochs)
steps_per_small_epoch = int(imgs_per_small_epoch / batch_size)

first_phase_train_reps = first_phase_epochs * saves_per_epoch
second_phase_train_reps = second_phase_epochs * saves_per_epoch
third_phase_train_reps = third_phase_epochs * saves_per_epoch

val_size = len(val_names_list)
val_imgs_per_rep = int(val_size / saves_per_epoch)
val_imgs_per_small_epoch = int(val_imgs_per_rep / small_epochs)
val_steps_per_small_epoch = int(val_imgs_per_small_epoch / batch_size) * 10


def add_noise(img):
    img /= 255.
    mean = 0.5
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, input_shape)
    noisy = img + gauss
    img = noisy.clip(0, 1)
    return img


def make_generators():
    global train_img_class_gen, val_img_class_gen
    image_class_generator_simple = ImageDataGenerator(samplewise_center=True,
                                                      samplewise_std_normalization=True,
                                                      rescale=1/255.)



    image_class_generator = ImageDataGenerator(samplewise_center=True,
                                               samplewise_std_normalization=True,
                                               rotation_range=30,
                                               width_shift_range=0.25,
                                               height_shift_range=0.25,
                                               zoom_range=0.3,
                                               horizontal_flip=True,
                                               preprocessing_function=add_noise)

    print('building image generators')
    train_img_class_gen = image_class_generator_simple.flow_from_directory(directory=train_class_images_path,
                                                                        target_size=input_shape[:2],
                                                                        batch_size=batch_size)

    val_img_class_gen = image_class_generator_simple.flow_from_directory(directory=val_class_images_path,
                                                                        target_size=input_shape[:2],
                                                                        batch_size=batch_size)
    print('Done building image generators')

make_generators()


def first_phase():
    global xcpetion_model
    tensorboard = TensorBoard(log_dir=first_phase_folder + 'tb_logs', batch_size=batch_size)
    # create the base pre-trained model
    input_tensor = Input(shape=input_shape)
    base_model = Xception(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # add a logistic layer
    predictions = Dense(classes_num, activation='softmax')(x)

    # this is the model we will train
    xcpetion_model = Model(inputs=base_model.input, outputs=predictions)

    for i in range(first_phase_train_reps):
        history = xcpetion_model.fit_generator(train_img_class_gen,
                                     steps_per_epoch=steps_per_small_epoch,
                                     epochs=small_epochs, verbose=1,
                                     validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
                                     workers=4, callbacks=[tensorboard])
        print(i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        xcpetion_model.save(first_phase_folder + str(ts) + '_xcpetion_model.h5')
        save_obj(history.history, str(ts) + '_xcpetion_history', folder=first_phase_folder)

    xcpetion_model.save(data_folder + '1st_phase_xcpetion_model.h5')


def second_phase():
    global xcpetion_model
    tensorboard = TensorBoard(log_dir=second_phase_folder + 'tb_logs', batch_size=batch_size)
    xcpetion_model = load_model(data_folder + '1st_phase_xcpetion_model.h5')

    trainable_layers_ratio = 1/3.0
    trainable_layers_index = int(len(xcpetion_model.layers) * (1 - trainable_layers_ratio))
    for layer in xcpetion_model.layers[:trainable_layers_index]:
       layer.trainable = False
    for layer in xcpetion_model.layers[trainable_layers_index:]:
       layer.trainable = True

    # for layer in xcpetion_model.layers:
    #     layer.trainable = True

    xcpetion_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

    # train the model on the new data for a few epochs
    for i in range(second_phase_train_reps):
        history = xcpetion_model.fit_generator(train_img_class_gen,
                                               steps_per_epoch=steps_per_small_epoch,
                                               epochs=small_epochs, verbose=1,
                                               validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
                                               workers=4, callbacks=[tensorboard])
        print(i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        xcpetion_model.save(second_phase_folder + str(ts) + '_xcpetion_model.h5')
        save_obj(history.history, str(ts) + '_xcpetion_history.h5', folder=second_phase_folder)

    xcpetion_model.save(data_folder + '2nd_phase_xcpetion_model.h5')


def third_phase():
    global xcpetion_model, new_xcpetion_model, optimizer
    # tensorboard = TensorBoard(log_dir=third_phase_folder + 'tb_logs', batch_size=batch_size)
    xcpetion_model = load_model(data_folder + '2nd_phase_xcpetion_model.h5')

    # add regularizers to the convolutional layers
    trainable_layers_ratio = 1 / 2.0
    trainable_layers_index = int(len(xcpetion_model.layers) * (1 - trainable_layers_ratio))
    for layer in xcpetion_model.layers[:trainable_layers_index]:
        layer.trainable = False
    for layer in xcpetion_model.layers[trainable_layers_index:]:
        layer.trainable = True

    for layer in xcpetion_model.layers:
        layer.trainable = True
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            layer.kernel_regularizer = regularizers.l2(0.001)
            layer.activity_regularizer = regularizers.l1(0.001)

    # add dropout and regularizer to the penultimate Dense layer
    predictions = xcpetion_model.layers[-1]
    dropout = Dropout(0.3)
    fc = xcpetion_model.layers[-2]
    fc.kernel_regularizer = regularizers.l2(0.001)
    fc.activity_regularizer = regularizers.l1(0.001)

    x = dropout(fc.output)
    predictors = predictions(x)
    new_xcpetion_model = Model(inputs=xcpetion_model.input, outputs=predictors)

    optimizer = Adam(lr=0.1234)
    start_lr = 0.001
    end_lr = 0.00001
    step_lr = (end_lr - start_lr) / (third_phase_train_reps - 1)
    new_xcpetion_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    for i in range(third_phase_train_reps):
        lr = start_lr + step_lr * i
        K.set_value(new_xcpetion_model.optimizer.lr, lr)
        print(i, 'out of ', third_phase_train_reps, '\nlearning rate ', K.eval(new_xcpetion_model.optimizer.lr))
        history = new_xcpetion_model.fit_generator(train_img_class_gen,
                                                   steps_per_epoch=steps_per_small_epoch,
                                                   epochs=small_epochs, verbose=1,
                                                   validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
                                                   workers=4, callbacks=[LosswiseKerasCallback(tag='keras xcpetion model')])
        print(i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        new_xcpetion_model.save(third_phase_folder + str(ts) + '_xcpetion_model.h5')
        save_obj(history.history, str(ts) + '_xcpetion_history.h5', folder=third_phase_folder)

    new_xcpetion_model.save(data_folder + '3rd_phase_xcpetion_model.h5')


def smoothed_third_phase():
    # reg_num = 0.001
    # drop_num = 0.3
    global xcpetion_model, new_xcpetion_model, optimizer

    # tensorboard = TensorBoard(log_dir=third_phase_folder + 'tb_logs', batch_size=batch_size)
    # xcpetion_model = load_model(data_folder + '2nd_phase_xcpetion_model.h5')
    #
    # predictions = xcpetion_model.layers[-1]
    # fc = xcpetion_model.layers[-2]
    # fc.name = 'penultimate_fc'
    #
    # x = Dropout(rate=drop_num, name='dropout_1')(fc.output)
    # predictors = predictions(x)
    # new_xcpetion_model = Model(inputs=xcpetion_model.input, outputs=predictors)
    #
    # for layer in new_xcpetion_model.layers[:trainable_layers_index]:
    #     layer.trainable = False
    # for layer in new_xcpetion_model.layers[trainable_layers_index:]:
    #     layer.trainable = True

    start_lr_num = 0.0001
    end_lr_num = 0.00005
    lr_nums = list(np.linspace(start_lr_num, end_lr_num, third_phase_train_reps))

    start_reg_num = 0.0001
    end_reg_num = 0.005
    reg_nums = list(np.linspace(start_reg_num, end_reg_num, third_phase_train_reps))
    # reg_num = 0.001

    start_drop_num = 0.05
    end_drop_num = 0.5
    drop_nums = list(np.linspace(start_drop_num, end_drop_num, third_phase_train_reps))
    # drop_num = 0.3

    for i in range(third_phase_train_reps):
        print('clear session')
        K.clear_session()
        print('load new model...')
        try:
            new_xcpetion_model = load_model(smoothed_third_phase_folder + str(ts) + '_xcpetion_model.h5')
            print('loaded model: ' + str(ts) + '_xcpetion_model.h5')
        except:
            new_xcpetion_model = load_model(initial_model)
            print('loaded model: ' + initial_model)

        trainable_layers_ratio = 1 / 3.0
        trainable_layers_index = int(len(new_xcpetion_model.layers) * (1 - trainable_layers_ratio))

        lr_num = lr_nums[i]
        reg_num = reg_nums[i]
        drop_num = drop_nums[i]
        print('lr: ', lr_num, '\nregularizer: ', reg_num, '\ndropout: ', drop_num)

        # add regularizers to the convolutional layers
        for layer in new_xcpetion_model.layers[trainable_layers_index:]:
            if isinstance(layer, keras.layers.convolutional.Conv2D) \
                    or isinstance(layer, keras.layers.Dense):
                layer.kernel_regularizer = regularizers.l2(reg_num)
                layer.activity_regularizer = regularizers.l1(reg_num)
            elif isinstance(layer, keras.layers.Dropout):
                layer.rate = drop_num

        optimizer = Adam(lr=lr_num)
        new_xcpetion_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

        history = new_xcpetion_model.fit_generator(train_img_class_gen,
                                                   steps_per_epoch=steps_per_small_epoch,  # steps_per_small_epoch
                                                   epochs=small_epochs, verbose=1,  # small_epochs
                                                   validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
                                                   workers=4, callbacks=[LosswiseKerasCallback(tag='keras xcpetion model')])
        print(i, ' out of ', third_phase_train_reps)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        new_xcpetion_model.save(smoothed_third_phase_folder + str(ts) + '_xcpetion_model.h5')
        save_obj(history.history, str(ts) + '_xcpetion_history.h5', folder=smoothed_third_phase_folder)

    new_xcpetion_model.save(data_folder + 'smoothed_3rd_phase_xcpetion_model.h5')


def second_second_phase():
    global xcpetion_model, new_xcpetion_model, drop_rate
    dropout_Callback = Dropout_Callback()
    tensorboard = TensorBoard(log_dir=second_phase_folder + 'tb_logs', batch_size=batch_size)

    xcpetion_model = load_model(data_folder + '2nd_phase_xcpetion_model.h5')

    # trainable_layers_ratio = 1/3.0
    # trainable_layers_index = int(len(xcpetion_model.layers) * (1 - trainable_layers_ratio))
    # for layer in xcpetion_model.layers[:trainable_layers_index]:
    #    layer.trainable = False
    # for layer in xcpetion_model.layers[trainable_layers_index:]:
    #    layer.trainable = True

    # add dropout and regularizer to the penultimate Dense layer

    start_drop_num = 0.00
    end_drop_num = 0.5
    drop_nums = list(np.linspace(start_drop_num, end_drop_num, second_phase_train_reps))

    predictions = xcpetion_model.layers[-1]
    dropout = My_Dropout(0.2, name='fc_1024_dropout', seed=0)
    fc = xcpetion_model.layers[-2]

    x = dropout(fc.output)
    predictors = predictions(x)
    new_xcpetion_model = Model(inputs=xcpetion_model.input, outputs=predictors)

    new_xcpetion_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

    # train the model on the new data for a few epochs
    for i in range(second_phase_train_reps):

        drop_rate = 0.8  # float(drop_nums[i])
        print(i + 1, 'out of ', third_phase_train_reps)

        history = new_xcpetion_model.fit_generator(train_img_class_gen,
                                               steps_per_epoch= steps_per_small_epoch,
                                               epochs=small_epochs, verbose=1,
                                               validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
                                               workers=4, callbacks=[tensorboard, dropout_Callback])
        print(i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        new_xcpetion_model.save(second_second_phase_folder + str(ts) + '_xcpetion_model.h5')
        save_obj(history.history, str(ts) + '_xcpetion_history.h5', folder=second_second_phase_folder)

        new_xcpetion_model.save(data_folder + '2nd_2nd_phase_xcpetion_model.h5')


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

def get_class_list(name_id_dict):
    class_list = []
    for id_list in name_id_dict.values():
        class_list.append(id_list)
    return class_list


def get_balanced_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: float(majority/count) for cls, count in counter.items()}


def my_loss(y_true, y_pred):
    loss = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    print('size of y_true: ', K.int_shape(y_true))
    print('size of y_pred: ', K.int_shape(y_pred))
    print('size of loss: ', K.int_shape(loss))
    return loss


def drop_ase_phase():
    reg_num = 0.01
    drop_ase_xcpetion_model = load_model(drop_ase_xcpetion_model_path)

    # for layer in drop_ase_xcpetion_model.layers:
    #     if not layer.trainable:
    #         continue
    #     if isinstance(layer, keras.layers.convolutional.Conv2D) \
    #             or isinstance(layer, keras.layers.Dense):
    #         layer.kernel_regularizer = regularizers.l2(reg_num)
    #         layer.activity_regularizer = regularizers.l1(reg_num)


    # print(new_xcpetion_model.summary())

    # new_input = Input(shape=input_shape)
    # input_Dropout = Dropout(0.2, name='input_Dropout', seed=0)(new_input)
    # new_xcpetion_model.layers[1] = new_xcpetion_model.layers[1](input_Dropout)

    # input_layer = new_xcpetion_model.layers[0]
    # assert type(input_layer) is keras.layers.InputLayer, 'input_layer is of type: ' + str(type(input_layer))
    # layer_2 = new_xcpetion_model.layers[1]
    # assert type(layer_2) is keras.layers.Conv2D, 'input_layer is of type: ' + str(type(input_layer))
    # input_Dropout = Dropout(0.2, name='input_Dropout', seed=0)
    # y = input_Dropout(input_layer.output)
    # layer_2 = layer_2(y)

    # predictions = new_xcpetion_model.layers[-1]
    # assert type(predictions) is keras.layers.Dense, 'predictions is of type: ' + str(type(predictions))
    #
    # fc_1024_Dropout = Dropout(0.2, name='fc_1024_Dropout', seed=0)
    # fc = new_xcpetion_model.layers[-3]
    # assert type(fc) is keras.layers.Dense, 'fc is of type: ' + str(type(fc))
    #
    # x = fc_1024_Dropout(fc.output)
    # predictors = predictions(x)
    # double_drop_hinge_xcpetion_model = Model(inputs=new_xcpetion_model.input, outputs=predictors)

    drop_ase_xcpetion_model.compile(optimizer=Adam(lr=0.0005), loss=my_loss, metrics=['acc'])
    print(drop_ase_xcpetion_model.summary())

    classes_weights = get_balanced_class_weights(get_class_list(train_name_id_dict))
    # train the model on the new data for a few epochs
    for i in range(second_phase_train_reps):
        tensorboard = TensorBoard(log_dir=drop_ase_xcpetion_phase_folder + 'tb_logs/{}'.format(time.time()),
                                  batch_size=batch_size)
        print(i + 1, 'out of ', third_phase_train_reps)

        history = drop_ase_xcpetion_model.fit_generator(train_img_class_gen,
                                                                 steps_per_epoch=steps_per_small_epoch,
                                                                 epochs=small_epochs, verbose=1,
                                                                 validation_data=val_img_class_gen,
                                                                 validation_steps=val_steps_per_small_epoch,
                                                                 #class_weight=classes_weights,
                                                                 workers=4, callbacks=[tensorboard])
        print(i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        drop_ase_xcpetion_model.save(drop_ase_xcpetion_phase_folder + str(ts) + '_xcpetion_model.h5')
        save_obj(history.history, str(ts) + '_xcpetion_history.h5', folder=drop_ase_xcpetion_phase_folder)

        drop_ase_xcpetion_model.save(data_folder + '2nd_2nd_phase_xcpetion_model.h5')


def load_model_from_file(path):
    # load json and create model
    with open(path, 'r') as json_file:
        model = model_from_json(json_file.read())
    return model


def set_reg_drop(model, reg_num, drop_rate):
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dropout):
            model.get_layer(layer.name).rate = drop_rate
        if not layer.trainable:
            continue
        if isinstance(layer, (keras.layers.convolutional.Conv2D, keras.layers.Dense)):
            model.get_layer(layer.name).kernel_regularizer = regularizers.l2(reg_num)
            model.get_layer(layer.name).activity_regularizer = regularizers.l1(reg_num)
    return model


def save_model_to_file(model, path):
    with open(path, "w") as json_file:
        json_file.write(model.to_json())


def average_precision(y_true, y_pred):
    loss = K.floatx(0)

    loss = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    print('size of y_true: ', K.int_shape(y_true))
    print('size of y_pred: ', K.int_shape(y_pred))
    print('size of loss: ', K.int_shape(loss))
    return loss


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


def square_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)


class GapCallback(keras.callbacks.Callback):
    def __init__(self, validation_generator, validation_steps):
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps

    def on_train_begin(self, logs={}):
        self.preds = []
        self.trues = []
        self.probs = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        for batch_index in range(self.validation_steps):
            features, y_true = next(self.validation_generator)
            y_pred = np.asarray(self.model.predict(features))
            print(y_pred.shape)

            y_true = np.argmax(y_true, -1)
            prob = np.max(y_pred, -1)
            y_pred = np.argmax(y_pred, -1)
            print(y_pred.shape)

            self.preds.extend(y_pred.tolist())
            self.trues.extend(y_true.tolist())
            self.probs.extend(prob.tolist())
            print(y_pred.shape)

            y_preds = np.array(self.preds)
            y_trues = np.array(self.trues)
            probs = np.array(self.probs)

            # y_preds = np.reshape(y_preds, (-1, y_preds.shape[-1]))
            # y_trues = np.reshape(y_trues, (-1, y_trues.shape[-1]))
            # probs = np.reshape(probs, (-1, probs.shape[-1]))
            print(y_true.shape, y_pred.shape, y_trues.shape, y_preds.shape)

            true_pos = [true_label == pred_label for true_label, pred_label in zip(y_trues, y_preds)]
            # print(true_pos)
            true_pos = [x for _, x in sorted(zip(probs, true_pos), reverse=True)]
            # print(true_pos)
            gap = 0
            for i in range(len(true_pos)):
                precision = sum(true_pos[:i + 1]) / len(true_pos[:i + 1])
                gap += precision if true_pos[i] else 0
            gap /= len(true_pos)
            print(gap)

            # _gap = label_ranking_average_precision_score(y_true=y_trues, y_score=y_preds)
            # print(_gap)
            return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def continue_second():
    reg_num = 0.00
    drop_rate = 0.0
    gap_callback = GapCallback(val_img_class_gen, val_steps_per_small_epoch)
    # new_xcpetion_model = load_model(data_folder + '2nd_phase_xcpetion_model.h5')
    # save_model_to_file(new_xcpetion_model, data_folder + 'xcpetion_model_.json')
    # xception_model.save_weights(data_folder + 'chackpoint_second_phase_xcpetion_model.h5')

    new_xcpetion_model = load_model_from_file(data_folder + 'xcpetion_model_dropout_1024.json')
    new_xcpetion_model = set_reg_drop(new_xcpetion_model, reg_num, drop_rate)

    # predictions = xcpetion_model.get_layer('dense_2')
    # dropout = Dropout(0.2, name='gap_2048_dropout', seed=0)
    # fc = xcpetion_model.get_layer('dense_1')
    # gap = xcpetion_model.get_layer('global_average_pooling2d_1')
    # fc_1024_dropout = xcpetion_model.get_layer('fc_1024_dropout')
    # x = dropout(gap.output)
    # x = fc(x)
    # x = fc_1024_dropout(x)
    # predictors = predictions(x)
    # new_xcpetion_model = Model(inputs=xcpetion_model.input, outputs=predictors)
    # save_model_to_file(new_xcpetion_model, data_folder + 'xcpetion_model_dropout_2048_1024.json')

    # new_xcpetion_model.save_weights(data_folder + '2nd_phase_xcpetion_weights.h5')
    print(new_xcpetion_model.summary())
    new_xcpetion_model = load_weights_from_file(new_xcpetion_model,
                                                data_folder + 'xcpetion_model_.json',
                                                data_folder + '2nd_phase_xcpetion_weights.h5')
    # new_xcpetion_model.load_weights(data_folder + '2nd_phase_xcpetion_weights.h5', by_name=False)

    new_xcpetion_model.compile(optimizer=Adam(lr=0.0002), loss=square_error, metrics=['acc', gap])  # categorical_crossentropy
    print('dropout rate: ', new_xcpetion_model.get_layer('fc_1024_dropout').rate)

    for i in range(second_phase_train_reps):
        print(i + 1, 'out of ', third_phase_train_reps)

        history = new_xcpetion_model.fit_generator(train_img_class_gen,
                                                   steps_per_epoch=steps_per_small_epoch,
                                                   epochs=small_epochs, verbose=1,
                                                   validation_data=val_img_class_gen,
                                                   validation_steps=val_steps_per_small_epoch,
                                                   workers=4,
                                                   callbacks=[LosswiseKerasCallback(tag='keras xcpetion model'),
                                                              gap_callback])
        print(i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        new_xcpetion_model.save_weights(continue_second_phase_folder + str(ts) + '_mse_xcpetion_model.h5')
        # new_xcpetion_model.save(continue_second_phase_folder + str(ts) + '_xcpetion_model.h5')
        save_obj(history.history, str(ts) + '_xcpetion_mse_history.h5', folder=continue_second_phase_folder)

    new_xcpetion_model.save_weights(data_folder + 'continue_second_phase_xcpetion_model.h5')

continue_second()
