from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from preprocessing import Data


class Vectorizer:
    def __init__(self, dataset):
        """Base class constructor. Note that dataset should be a Dataset object."""
        self.dataset = dataset

    def vectorize(X):
        raise NotImplementedError()


class TFIDFWordVectorizer(Vectorizer):
    def __init__(self, dataset, ngram_range=(1, 2)):
        super().__init__(dataset)
        self.params = {
            "analyzer": "word",
            "tokenizer": dataset._tokenize,
            "stop_words": "english",
            "ngram_range": ngram_range,
            "max_df": 0.9,
            "min_df": 0.001,
            "max_features": None,
            "smooth_idf": True,
            "sublinear_tf": False,
        }
        self.vectorizer = TfidfVectorizer(**self.params)
        self.vectors = None
        self.vocab = None

    def vectorize(self, sparse=True):
        term_doc_matrix = self.vectorizer.fit_transform(self.dataset.raw_tweets)
        self.vocab = self.vectorizer.vocabulary_
        if not sparse:
            self.vectors = term_doc_matrix.toarray()
        else:
            self.vectors = term_doc_matrix
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


class CharEmbeddings(Vectorizer):
    def __init__(self, dataset):
        super().__init__(dataset)

    def parse(self):
        vectors = {}
        embeddings_path = 'Data/glove.840B.300d-char.txt'
        with open(embeddings_path, 'r') as f:
            for row in f:
                split_row = row.strip().split(" ")
                vector = np.array(split_row[1:], dtype=float)
                char = split_row[0]
                vectors[char] = vector
        print(len(vectors))


if __name__ == "__main__":
    DATAFILE = "./Data/twitter_hate.csv"
    D = Data(DATAFILE, preprocess=False)

    word_V = TFIDFWordVectorizer(D)
    word_X = word_V.vectorize()
    print('Word TFIDF shape: ', word_X.shape)

    # ngram_range = (3,3)
    # char_V = TFIDFCharVectorizer(D, ngram_range)
    # char_X = char_V.vectorize()
    # print('Char TFIDF shape: ', char_X.shape)

    # emb_V = CharEmbeddings(D)
    # emb_V.parse()
    pass
