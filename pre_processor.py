import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn import svm
from sklearn.model_selection import train_test_split


def get_samples_index(y, num_per_label=1000):
    label_indices = {}
    unique_labels = np.unique(y)
    for label in unique_labels:
        label_indices[label] = []
    for index in range(len(y)):
        label_indices[y[index]].append(index)

    selected_indices = []
    for label in unique_labels:
        print("label {} has {}".format(label, len(label_indices[label])))
        if len(label_indices[label]) >= num_per_label:
            replace = False
        else:
            replace = True
        select_index = np.random.choice(label_indices[label], num_per_label, replace=replace)
        selected_indices = selected_indices + list(select_index)
    return sorted(selected_indices)


def feature_selection(X, y):
    info_gain = mutual_info_classif(X, y)
    info_gain_indices = list(np.argsort(info_gain)[::-1])
    best_score = 0.0
    max_info_size = 0
    for i in range(len(info_gain_indices)):
        X_sample = X[:, info_gain_indices[:i+1]]
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y, test_size=0.2, random_state=0)
        clf = svm.SVC(C=1.0, kernel='rbf')
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if score > best_score:
            best_score = score
            max_info_size = i + 1
    print("best score {}, max inforation size {}".format(best_score, max_info_size))
    return info_gain_indices[:max_info_size]







