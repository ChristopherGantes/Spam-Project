# models/svm.py
from sklearn.svm import SVC


class SVMModel:
    def __init__(self):
        self.model = SVC()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
