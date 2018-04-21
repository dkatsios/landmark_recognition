import numpy as np
import pickle


working_folder = r'E:\Data_Files\Workspaces\PyCharm\kaggle\landmark-recognition-challenge\\'
data_folder = working_folder + r'data\\'
results_file = data_folder + 'test_results.csv'


def load_obj(name, folder=data_folder):
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)

val_name_id_dict = load_obj('val_name_id_dict')

true_labels = []
pred_labels = []
probs = []

with open(results_file, 'r') as f:
    f.readline()
    for line in f:
        tokens = line.strip().split(',')
        name = tokens[0]

        pred_label, prob = tokens[1].split(' ')
        pred_label = str(pred_label)
        prob = float(prob)
        true_label = str(val_name_id_dict[name])

        true_labels.append(true_label)
        pred_labels.append(pred_label)
        probs.append(prob)

true_pos = [true_label == pred_label for true_label, pred_label in zip(true_labels, pred_labels)]
# print(true_pos)
true_pos = [x for _, x in sorted(zip(probs,true_pos), reverse=True)]
# print(true_pos)
gap = 0
for i in range(len(true_pos) // 2):
    precision = sum(true_pos[:i+1]) / len(true_pos[:i+1])
    gap += precision if true_pos[i] else 0
gap /= len(true_pos)
print(gap)
print(sum(true_pos))
