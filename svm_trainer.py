from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm


def train_svm(data, labels, C=1.0):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    clf = svm.SVC(C=C, kernel='rbf', gamma='auto')
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("SVM score is {}, C = {}".format(score, C))
