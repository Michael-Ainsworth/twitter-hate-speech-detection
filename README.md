# twitter-hate-speech-detection

### Project Description

The motivation for this project is to build a reliable pipeline that can identify and protect twitter users from hateful speech. To achieve this goal, we will experiment with various tweet vectorization techniques and evaluate performance on a logistic regression model, a random forest model, and a neural network.

To download the pre-trained GloVe embeddings, visit this link: https://nlp.stanford.edu/projects/glove/
Go to the "Download pre-trained word vectors" section and select the download you want. For this project, we use glove.6b.zip (trained on Wikipedia 2014 + Gigaword 5).

Resources: https://github.com/minimaxir/char-embeddings (for obtaining pre-trained character embeddings), https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89 (used for structuring our NN architecture and code)



The files in this repository are not meant to be run directly. Rather, you should write your own driver program and import all the classes as necessary. The following is an overview of the file and class structure:

`preprocessing.py` - contains the Data class used for storing, preprocessing, augmenting, and balancing the dataset. Import it like so: `from preprocessing import Data`. To instantiate, you must provide the path to the twitter_hate.csv file.

`vectorize.py`- contains 4 classes, one for each vectorizer: TFIDFWordVectorizer, TFIDFCharVectorizer, CharEmbeddings, and DocEmbeddings. When instantiating these classes, you must provide a Data object.

`models.py` - contains 