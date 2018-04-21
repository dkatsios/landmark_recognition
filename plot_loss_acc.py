from os import listdir
import pickle
import matplotlib.pyplot as plt
import numpy as np

working_folder = r'E:\Data_Files\Workspaces\PyCharm\kaggle\landmark-recognition-challenge/'
data_folder = working_folder + r'data/'

history_dir = data_folder + 'siamese_first_phase_folder/'


def load_obj(name, folder=data_folder):
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_histories_lists(history_dir, suffix='.pkl'):
    categories = ['loss', 'acc', 'val_loss', 'val_acc']
    dicts = dict()
    for category in categories:
        dicts[category] = []
    history_files = sorted(list(listdir(history_dir)))
    for file in history_files:
        if suffix in file:
            history = load_obj(file.replace(suffix, ''), folder=history_dir)
            for category in categories:
                dicts[category].extend(history[category])
    return dicts


def plot_history(dicts):
    window = 20
    val_acc = np.asarray(dicts['val_acc'])
    val_acc = np.convolve(val_acc, np.ones((window,)) / window, mode='valid')

    val_loss = np.asarray(dicts['val_loss'])
    val_loss = np.convolve(val_loss, np.ones((window,)) / window, mode='valid')

    # summarize history for accuracy
    plt.plot(dicts['acc'])
    # plt.plot(dicts['val_acc'])
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'smooth test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(dicts['loss'])
    # plt.plot(dicts['val_loss'])
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'smooth test'], loc='upper left')
    plt.show()

history_dicts = get_histories_lists(history_dir)
plot_history(history_dicts)



