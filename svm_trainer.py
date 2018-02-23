from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import pdb


def train_svm(train_data, train_labels, test_data, test_labels, c=1.0, kernel="rbf", g=0.01):
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
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print("SVM score is {}".format(score))
    print("confusion matrix: ")
    print(cnf_matrix)
    return score, clf
