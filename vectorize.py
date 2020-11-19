from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import Dataset

class Vectorizer:
    def __init__(self, dataset):
        """Base class constructor. Note that dataset should be a Dataset object."""
        self.dataset = dataset

    def vectorize(X):
        raise NotImplementedError()

class TFIDFWordVectorizer(Vectorizer):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.params = {
            "analyzer": "word",
            "tokenizer": dataset._tokenize,
            "stop_words": "english",
            "max_features": None,
            "smooth_idf": True,
            "sublinear_tf": False,
        }
        self.vectorizer = TfidfVectorizer(**self.params)
        self.vectors = None

    def vectorize(self):
        term_doc_matrix = self.vectorizer.fit_transform(self.dataset.raw_tweets)
        self.vectors = term_doc_matrix.toarray()
        return self.vectors

if __name__ == "__main__":
    DATAFILE = "./Data/twitter_hate.csv"
    D = Dataset(DATAFILE, preprocess=False)
    V = TFIDFWordVectorizer(D)
    #print(V.vectorizer)

    X = V.vectorize()
    print(X.shape)
    pass
