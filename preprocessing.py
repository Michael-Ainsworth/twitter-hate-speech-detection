import copy
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
from nltk.tokenize import wordpunct_tokenize

from tqdm import tqdm

class Data:
    """
        Class to load, pre-process, and store a Twitter dataset.

        Note: Right now, this class is explicitly tailored to process the twiter_hate.csv file.
    """
    def __init__(self, filepath, tweet_col=-1, label_col=-2, preprocess=True):
        self.escaped = {"&amp;": "&", "&gt;": ">", "&lt;": "<", "&apos;": "\'", "&quot;": '\"', "&#124;": "|", "&#91;": "[", "&#93;": "]"}
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'ou', "ou're", "ou've", "ou'll", "ou'd", 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'I', "'"]
        self.tweet_col = tweet_col
        self.label_col = label_col
        self.raw_tweets, self.labels = self._load_data(filepath, preprocess)
        self.tweets = None
        self.augmentations = 0
        self.train_tweets = None
        self.train_labels = None
        self.test_tweets = None
        self.test_labels = None

        # Pre-process tweets
        if preprocess:
            self.tweets = [self._preprocess_tweet(t, tokenize=True, remove_stopwords=True) for t in self.raw_tweets]

    def _train_test_split(self, seed):
        X_train, X_test, y_train, y_test = train_test_split(np.array(self.tweets), np.array(self.labels), test_size=.2, random_state=seed)
        self.train_tweets = X_train.tolist()
        self.test_tweets = X_test.tolist()
        self.train_labels = y_train.tolist()
        self.test_labels = y_test.tolist()

    def _split_row(self, row, num_cols):
        """Given a main row of the file, split on the comma."""
        split_row = []
        while len(split_row) < num_cols - 1:
            idx = row.index(",")
            split_row.append(row[:idx])
            row = row[idx + 1:]
        split_row.append(row[1:-1]) # add the tweet to the split row, but remove the first and last quotation mark
        return split_row

    def _load_data(self, filename, preprocess, header=True, num_cols=7):
        # TODO: We can just read this in with pandas. This does the same thing and was a waste of time.

        # Read data from file
        rows = []
        start_idx = 1 if header else 0
        with open(filename, "r") as f:
            lines = [l.strip() for l in f.readlines()][start_idx:] # ignore header row
            for i in range(len(lines)):
                match = re.search(r"^\d+\,(\d,){5}", lines[i])
                if match: # if we're at a new example
                    rows.append(lines[i])
                else: # else, this line is part of the tweet in the current example
                    rows[-1] += "\n" + lines[i]

        # Convert rows into 2-d list of rows split on commas
        rows = [self._split_row(r, num_cols) for r in rows]

        return [r[self.tweet_col] for r in rows], [r[self.label_col] for r in rows]

    def _preprocess_tweet(self, tweet, tokenize=False, remove_stopwords=False):
        tweet = self._remove_call_outs(tweet)
        tweet = self._fix_escaped_tokens(tweet) # map escaped tokens to a human-readable format (e.g. &amp; --> &)
        if tokenize:
            #print('Tokenizing...')
            tweet = self._tokenize(tweet)

        if remove_stopwords:
            tweet = self._remove_stopwords(tweet)

        tweet = [t for t in tweet if t] # remove blank spaces
        return tweet
    def _TFIDF_doc_tokenize(self, tweet):
        return list(tweet.split(' '))

    def _tokenize(self, tweet):
        """Split a tweet into lowercase tokens (including punctuation)."""
        tokens = wordpunct_tokenize(tweet)
        clean_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "<&#":
                clean_tokens.append(tokens[i] + tokens[i + 1] + tokens[i + 2])
                i += 3
            else:
                clean_tokens.append(tokens[i].lower())
                i += 1
        return clean_tokens

    def _remove_stopwords(self, tweet):
        """Remove stopwords in a list of tokens."""
        return [t for t in tweet if t not in self.stop_words]

    def _remove_call_outs(self, tweet):
        """Remove any "@person" token."""
        ats = re.findall(r"\@\w+\:*", tweet)
        for a in ats:
            tweet = tweet.replace(a, "")
        return tweet

    def _fix_escaped_tokens(self, tweet):
        """Replace standard escaped tokens with human-readable versions and surround emojis with <>."""
        esc = re.findall(r"\&\#*\w+\;*", tweet)
        for e in esc:
            tweet = tweet.replace(e, self.escaped.get(e, "")) # replace with nothing for now. Maybe add this back later: f"<{e}>"
        return tweet

    def augment_data(self):
        # TODO: Implement this
        new_tweets = []
        new_labels = []
        for tweet, label in tqdm(zip(self.tweets, self.labels), total=len(self.labels)):
            if int(label) == 0:
                results = self._augment_single_tweet(tweet)
                new_tweets.extend(results)
                new_labels.extend([label for _ in range(len(results))])
                self.augmentations += len(results)

        self.tweets.extend(new_tweets)
        self.labels.extend(new_labels)
        return # TODO: return something

    def _candidate_to_replace(self, word, cand):
        """Determine if a candidate is sufficient to replace the chosen word. Must be a distinct (non-homonym) unigram."""
        distinct = (word not in cand[:cand.find(".")])
        one_word = "_" not in cand
        return distinct and one_word

    def _candidate_to_be_replaced(self, token, tag):
        """
        Determine if a word is eligible to be replaced.
        Must be a noun/adjective/verb, contain strictly letters, and be 3+ letters long.
        """
        noun = tag in ("NN", "NNS")
        verb = tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")
        adjective = tag in ("JJ", "JJR", "JJS")
        letters_only = token.isalpha()
        length = len(token) > 2
        return (noun or verb or adjective) and letters_only and length

    def _augment_single_tweet(self, tweet):
        """
        Augment a single data example by selecting and replacing 1 noun, if possible.
        Helpful resource for NLTK POS tagging: https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/
        """
        #print(tweet)
        #print()
        pos_map = {"NN": "n", "NNS": "n", "VB": "v", "VBD": "v", "VBG": "v", "VBN": "v", "VBP": "v", "VBZ": "v", "JJ": "a", "JJR": "a", "JJS": "a"}

        pos_tags = nltk.pos_tag(tweet)
        to_replace = [(i, token, tag) for i, (token, tag) in enumerate(pos_tags) if self._candidate_to_be_replaced(token, tag)]

        if len(to_replace) == 0:
            return []

        # Find a synonym for each word (if possible)
        new_data = []
        for idx, word, tag in to_replace:
            #print(idx, word, tag, pos_map[tag])

            # Generate list of candidate replacements with WordNet and choose the first one
            candidates = [l for l in wn.synsets(word, pos=pos_map[tag]) if self._candidate_to_replace(word, l.name())]
            if len(candidates) > 0:
                r = candidates[0].name() # TODO: make this step smarter with word embeddings
                replacement = r[:r.find(".")]
                #print(replacement)

                # Make the replacement
                new_tweet = copy.deepcopy(tweet)
                new_tweet[idx] = replacement
                #print(new_tweet)
                #print()
                new_data.append(new_tweet)
            #else:
                #print("No synonym found\n")

        return new_data


if __name__ == "__main__":
    DATAFILE = "./Data/twitter_hate.csv"
    D = Data(DATAFILE, preprocess=True)
    D.augment_data()
    print(len(D.tweets))
    #D._train_test_split(1)

    #print(D.raw_tweets[0])
    #print(type(D.tweets[0]))
    # quit()
    #print(len(D.tweets), len(D.labels))
    #print(D.tweets[0])
