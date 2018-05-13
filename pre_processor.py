import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn import svm
from sklearn.model_selection import train_test_split
import bisect
import pdb


def get_samples_index(y, split = 0.2):
    label_indices = {}
    unique_labels = np.unique(y)
    for label in unique_labels:
        label_indices[label] = []
    for index in range(len(y)):
        label_indices[y[index]].append(index)

    for label in unique_labels:
        print("label {} has {}".format(label, len(label_indices[label])))

    # find the imbalanced class    
    nonzero_indices = []
    for label in unique_labels:
        if label != 0:
            nonzero_indices = nonzero_indices + label_indices[label]
    nonzero_indices = sorted(nonzero_indices)

    selected_train_indices_nonzero = nonzero_indices[:int((1-split)*len(nonzero_indices))]
    selected_test_indices_nonzero = nonzero_indices[int((1-split)*len(nonzero_indices)):]

    idx = bisect.bisect_left(label_indices[0], int((selected_train_indices_nonzero[-1] + selected_test_indices_nonzero[0])/2))
    train_indices_zero = label_indices[0][:idx]
    test_indices_zero = label_indices[0][idx:]

    # selected_train_indices_zero = np.random.choice(train_indices_zero, len(selected_train_indices_nonzero), replace=False)
    selected_train_indices_zero = [train_indices_zero[i] for i in np.linspace(0, len(train_indices_zero), num=len(selected_train_indices_nonzero)/2, endpoint=False).astype(int)]
    # selected_test_indices_zero = np.random.choice(test_indices_zero, len(selected_test_indices_nonzero), replace=False)
    selected_test_indices_zero = [test_indices_zero[i] for i in np.linspace(0, len(test_indices_zero), num=len(selected_test_indices_nonzero)/2, endpoint=False).astype(int)]

    return sorted(np.concatenate((selected_train_indices_zero, selected_train_indices_nonzero), axis=0)), sorted(np.concatenate((selected_test_indices_zero, selected_test_indices_nonzero), axis=0)), label_indices[0][idx]


def feature_selection(X, y):
    info_gain = mutual_info_classif(X, y)
    info_gain_indices = list(np.argsort(info_gain)[::-1])
    best_score = 0.0
    max_info_size = 0
    for i in range(len(info_gain_indices)):
        X_sample = X[:, info_gain_indices[:i+1]]
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y, test_size=0.2, shuffle=False)
        clf = svm.SVC(C=1.0, kernel='rbf')
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if score > best_score:
            best_score = score
            max_info_size = i + 1
    print("best score {}, max inforamtion size {}".format(best_score, max_info_size))
    return info_gain_indices[:max_info_size]







