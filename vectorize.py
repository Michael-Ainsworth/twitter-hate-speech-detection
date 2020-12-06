from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
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
    def __init__(self, dataset, emb_path, emb_dim=300):
        super().__init__(dataset)
        self.emb_dim = emb_dim
        self.embeddings = self.parse(emb_path)

    def parse(self, path):
        # return embeddings
        embeddings = {}
        embeddings_path = f'{path}/glove.840B.300d-char.txt'
        with open(embeddings_path, 'r') as f:
            for row in f:
                split_row = row.strip().split(" ")
                vector = np.array(split_row[1:], dtype=float)
                char = split_row[0]
                embeddings[char] = vector[:self.emb_dim]
        
        return embeddings

    def _make_vector(self, tweet):
        components = []

        for char in tweet:
            
            if char in self.embeddings:
                components.append(self.embeddings[char])

        # Create matrix
        doc_matrix = np.array(components)

        # Calculate average embedding for this document
        avg = np.sum(doc_matrix, axis=0) / doc_matrix.shape[0]

        return avg

    def vectorize(self):
        # Vectorize each tweet
        vecs = [self._make_vector(''.join(tweet)) for tweet in self.dataset.tweets]
        
        # Create matrix with all valid document embeddings
        self.vectors = np.stack(vecs, axis=0)

        return self.vectors


class DocEmbeddings(Vectorizer):
    def __init__(self, dataset, *, scheme="unweighted", emb_path, emb_dim):
        super().__init__(dataset)
        self.emb_dim = emb_dim
        self.scheme = scheme

        self.embeddings = self.load_word_embeddings(emb_path) # can add limit argument to make development a bit faster

        # Some counters to compute metrics for the vectorization
        self.total_tokens = 0
        self.successful_replacements = 0

    def load_word_embeddings(self, path, limit=-1):
        file_name = f"{path}/glove.6B.{self.emb_dim}d.txt"
        print(f"Loading GloVe word embedding from {file_name}")
        embeddings = {}
        with open(file_name, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == limit:
                     break
                line = line.strip().split(" ") # the first token on each line is the word, followed by the vector
                if line[0].lower() not in embeddings:
                    embeddings[line[0]] = [float(x) for x in line[1:]]

        return embeddings

    def _make_vector(self, tweet):
        #print(tweet, len(tweet))
        components = []
        for token in tweet:
            self.total_tokens += 1
            # Try to do a direct lookup first, so we can maybe make life easy and catch punctuation
            if token in self.embeddings:
                components.append(self.embeddings[token])
                self.successful_replacements += 1

        # Early stopping if we couldn't find anything for this
        if len(components) == 0:
            return None

        # Create matrix
        doc_matrix = np.array(components)

        # Calculate average embedding for this document
        if self.scheme == "unweighted":
            avg = np.sum(doc_matrix, axis=0) / doc_matrix.shape[0]
            #avg = avg.reshape(-1, 1).T
        elif self.scheme == "tfidf":
            # TODO: Weight each word embedding by its TFIDF score?
            raise NotImplementedError()
        else:
            raise ValueError("Invalid value for scheme argument")

        return avg

    def vectorize(self, success_rate=False):
        # Vectorize each tweet
        vecs = [self._make_vector(tweet) for tweet in self.dataset.tweets]

        # Some tweets can't be vectorized, so drop those
        reduced_vecs = [v for v in vecs if v is not None]

        # Create matrix with all valid document embeddings
        self.vectors = np.stack(reduced_vecs, axis=0)

        # Calculate success rate
        if success_rate:
            sr = round(100 * self.successful_replacements / self.total_tokens, 2)
            print(f"Found embeddings for {sr}% of all possible tokens")

        return self.vectors


if __name__ == "__main__":
    DATAFILE = "./Data/twitter_hate.csv"
    D = Data(DATAFILE, preprocess=True)

    # DE = DocEmbeddings(D, emb_path="./Data", emb_dim=300)
    # doc_vecs = DE.vectorize(success_rate=True)
    # print(doc_vecs.shape)

    # word_V = TFIDFWordVectorizer(D)
    # word_X = word_V.vectorize()
    # print('Word TFIDF shape: ', word_X.shape)


    # ngram_range = (3,3)
    # char_V = TFIDFCharVectorizer(D, ngram_range)
    # char_X = char_V.vectorize()
    # print('Char TFIDF shape: ', char_X.shape)


    emb_V = CharEmbeddings(dataset=D, emb_path='./Data', emb_dim=5)
    # emb_V.parse(path='./Data')
    # emb_V.test()
    vecs = emb_V.vectorize()
    # emb_V
    pass
