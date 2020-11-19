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

class TFIDFCharVectorizer(Vectorizer):
    def __init__(self, dataset, ngram_range):
        super().__init__(dataset)
        self.params = {
            "analyzer": "char",
            "tokenizer": dataset._tokenize,
            "stop_words": None,
            "ngram_range": ngram_range,
            "max_df": 0.9,
            "min_df": 0.001,
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

    word_V = TFIDFWordVectorizer(D)
    word_X = word_V.vectorize()
    print('Word TFIDF shape: ', word_X.shape)

    ngram_range = (3,3)
    char_V = TFIDFCharVectorizer(D, ngram_range)
    char_X = char_V.vectorize()
    print('Char TFIDF shape: ', char_X.shape)

    pass
