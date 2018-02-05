from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def train_svm(data, labels, c=1.0, kernel="rbf"):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    clf = svm.SVC(C=c, kernel=kernel, gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print("SVM score is {}".format(score))
    print("confusion matrix: ")
    print(cnf_matrix)
    print()
