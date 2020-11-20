from preprocessing import Dataset
from vectorize import TFIDFCharVectorizer

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix


class LogisticRegressionModel():
    def __init__(self):
        self.params = {
            "penalty": 'l2',
            "verbose": 0
        }
        self.model = LogisticRegression(**self.params)
        self.score = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        # return clf
    
    def predict(self, X_test, y_test):
        self.model.predict_proba(X_test)
        self.score = self.model.score(X_test, y_test)
        preds = self.model.predict(X_test)
        return self.score, preds


if __name__ == "__main__":
    DATAFILE = "./Data/twitter_hate.csv"
    D = Dataset(DATAFILE, preprocess=False)

    ngram_range = (2,2)
    char_V = TFIDFCharVectorizer(D, ngram_range)
    char_X = char_V.vectorize()
    print('Char TFIDF shape: ', char_X.shape)

    labels = np.array(D.labels).astype(float)
    labels = np.where(labels < 0.5, 1, 0)
    print('Label shape: ', labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(char_X, labels, test_size=0.33, random_state=42)

    m = LogisticRegressionModel()
    m.fit(X_train, y_train)
    score, preds = m.predict(X_test, y_test)
    print('Score: ', score)

    print(confusion_matrix(preds,y_test))

    pass