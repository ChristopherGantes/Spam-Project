# models/naive_bayes.py

from sklearn.naive_bayes import MultinomialNB


class NaiveBayesModel:
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
