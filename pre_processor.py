import numpy as np


def get_samples_index(y, num_per_label=1000):
    label_indice = {}
    unique_labels = np.unique(y)
    for label in unique_labels:
        label_indice[label] = []
    for index in range(len(y)):
        label_indice[y[index]].append(index)

    selected_indice = []
    for label in unique_labels:
        print("label {} has {}".format(label, len(label_indice[label])))
        if len(label_indice[label]) >= num_per_label:
            replace = False
        else:
            replace = True
        select_index = np.random.choice(label_indice[label], num_per_label, replace=replace)
        selected_indice = selected_indice + list(select_index)
    return sorted(selected_indice)

