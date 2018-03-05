from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import numpy as np
import pdb


def train_svm(train_data, train_labels, test_data, test_labels, c=1.0, kernel="rbf", g=0.01, threshold=True):
    X_train = train_data
    X_test = test_data
    y_train = train_labels
    y_test = test_labels
    # data preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    clf = svm.SVC(C=c, kernel=kernel, gamma=g)
    clf.fit(X_train, y_train)
    if threshold:
        scores = clf.decision_function(X_test)
        positive_scores = [max(score, 0) for score in scores]
        threshold = np.mean(np.array(positive_scores))
        y_pred = [int(score > threshold) for score in scores]
    else:
        y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print("SVM accuracy is {}".format(accuracy))
    print("SVM precision is {}".format(precision))
    print("confusion matrix: ")
    print(cnf_matrix)
    return accuracy, clf
