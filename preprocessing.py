import pandas as pd
import re

from nltk.tokenize import wordpunct_tokenize

class Dataset:
    """
        Class to load, pre-process, and store a Twitter dataset.

        Note: Right now, this class is explicitly tailored to process the twiter_hate.csv file.
    """
    def __init__(self, filepath, tweet_col=-1, label_col=-2, preprocess=True):
        self.escaped = {"&amp;": "&", "&gt;": ">", "&lt;": "<", "&apos;": "\'", "&quot;": '\"', "&#124;": "|", "&#91;": "[", "&#93;": "]"}
        self.tweet_col = tweet_col
        self.label_col = label_col
        self.raw_tweets, self.labels = self._load_data(filepath, preprocess)
        self.tweets = None

        # Pre-process tweets
        if preprocess:
            self.tweets = [self._preprocess_tweet(t, tokenize=True, remove_stopwords=True) for t in self.raw_tweets]



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
            # print('Tokenizing...')
            tweet = self._tokenize(tweet)
        if remove_stopwords:
            tweet = self._remove_stopwords(tweet)
        return tweet

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
        # TODO: Implement this
        return tweet

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
            tweet = tweet.replace(e, self.escaped.get(e, f"<{e}>"))
        return tweet

    def _augment_data():
        # TODO: Implement this
        pass

if __name__ == "__main__":
    DATAFILE = "./Data/twitter_hate.csv"
    D = Dataset(DATAFILE, preprocess=True)
    print(len(D.tweets))
    print(D.raw_tweets[0])
    print(type(D.tweets[0]))
    # quit()
    print(len(D.tweets), len(D.labels))
    print(D.tweets[0])
