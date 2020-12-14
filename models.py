from preprocessing import Data
from vectorize import TFIDFCharVectorizer, TFIDFWordVectorizer, CharEmbeddings, DocEmbeddings
from metrics import binary_metrics


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix



class LogisticRegressionModel():
    def __init__(self):

        #initialize score
        self.score = None

        #-----COMMENT OUT THIS LINE WHEN USING HYPERPARAMETER TUNING-----#
        self.model = LogisticRegression()
        #----------------------------------------------------------------#

        """
        ############### HYPERPARAMETER SEARCH  ###############
        #create base model for search
        self.model_base = LogisticRegression()

        # Set Regularizer Search Vector
        self.lr_C = [0.00001,0.0001,0.001,0.01,0.1,1,10,100]

        #Set Intercept Search Vector
        self.lr_fit_intercept = [True,False]

        #Set Solver Vector
        self.lr_solver = ['lbfgs','sag','saga']

        #Create Hyperparameter Grid as Dictionary
        self.grid = {'C': self.lr_C,
                     'fit_intercept': self.lr_fit_intercept,
                     'solver': self.lr_solver}

        #Create exhaustive search estimator
        self.model = GridSearchCV(estimator = self.model_base, param_grid = self.grid, cv = 5, verbose = 0, n_jobs = -1)
        ########## END HYPERPARAMETER SEARCH  ##########
        """

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        #print(self.model.best_params_)
        # return clf

    def predict(self, X_test, y_test):
        raw_preds = self.model.predict_proba(X_test)
        self.score = self.model.score(X_test, y_test)
        preds = self.model.predict(X_test)
        return self.score, preds, raw_preds[:,1]


class RandomForestModel():
    def __init__(self):

        #initialize score
        self.score = None

        #-----COMMENT OUT THIS LINE WHEN USING HYPERPARAMETER TUNING-----#
        self.model = RandomForestClassifier()
        #----------------------------------------------------------------#

        """
        ############### HYPERPARAMETER SEARCH  ###############
        #create base model for search
        self.model_base = RandomForestClassifier()

        # Set Estimator Search Vector
        self.rf_n_estimators = [int(x) for x in np.linspace(200,1000,5)]
        self.rf_n_estimators.append(1500)
        self.rf_n_estimators.append(2000)

        #Set Depth Search Vector
        self.rf_max_depth = [int(x) for x in np.linspace(5,55,11)]
        self.rf_max_depth.append(None)

        #Set Criterion Vector
        self.rf_criterion = ['gini', 'entropy']

        #set Min Sample Split Vector
        self.rf_min_samples_split = [int(x) for x in np.linspace(2,10,9)]

        #set Max_Features
        self.rf_max_features = ['auto', 'sqrt', 'log2']
        #Set Impurity Vector
        self.rf_min_impurity_decrease = [0.0,0.05,0.1]

        #Set Bootstrap
        self.rf_bootstrap = [True, False]

        #Create Hyperparameter Grid as Dictionary
        self.grid = {'n_estimators': self.rf_n_estimators,
                     'max_depth': self.rf_max_depth,
                     'criterion': self.rf_criterion,
                     'min_samples_split': self.rf_min_samples_split,
                     'max_features': self.rf_max_features,
                     'min_impurity_decrease': self.rf_min_impurity_decrease,
                     'bootstrap': self.rf_bootstrap}

        #Create random search estimator
        self.model = RandomizedSearchCV(estimator = self.model_base, param_distributions = self.grid, n_iter = 100, cv = 5, verbose = 2, random_state = 42, n_jobs = -1)
        ########## END HYPERPARAMETER SEARCH  ##########
        """

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        #print(self.model.best_params_)

    def predict(self, X_test, y_test):
        raw_preds = self.model.predict_proba(X_test)
        self.score = self.model.score(X_test, y_test)
        preds = self.model.predict(X_test)
        return self.score, preds, raw_preds[:,1]

class SVMModel():
    def __init__(self):

        #initialize score
        self.score = None

        #-----COMMENT OUT THIS LINE WHEN USING HYPERPARAMETER TUNING-----#
        self.model = svm.SVC()
        #----------------------------------------------------------------#

        """
        ############### HYPERPARAMETER SEARCH  ###############
        #Create base model for search
        self.model_base = svm.SVC()

        # Set C Search Vector
        self.svc_C = [0.00001,0.0001,0.001,0.01,0.1,1,10,100]

        #Set Kernel Search Vector
        self.svc_kernel= ['rbf']

        #Set Gamma Vector
        self.svc_gamma = ['scale', 'auto']

        #set Probability Vector
        self.svc_probability = [True]

        #set Random State
        self.svc_random_state = [int(x) for x in np.linspace(0,50,5)]

        #Create Hyperparameter Grid as Dictionary
        self.grid = {'C': self.svc_C,
                     'kernel': self.svc_kernel,
                     'gamma': self.svc_gamma,
                     'probability': self.svc.probability,
                     'random_state': self.svc.random_state}

        #Create grid search estimator
        self.model = GridSearchCV(estimator = self.model_base, param_grid = self.grid, cv = 5, verbose = 0, random_state = 42, n_jobs = -1)
        ########## END HYPERPARAMETER SEARCH  ##########
        """

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        #print(self.model.best_params_)

    def predict(self, X_test, y_test):
        raw_preds = self.model.predict_proba(X_test)
        self.score = self.model.score(X_test, y_test)
        preds = self.model.predict(X_test)
        return self.score, preds, raw_preds[:,1]

class AdaBoostModel():
    def __init__(self):

        #initialize score
        self.score = None

        #-----COMMENT OUT THIS LINE WHEN USING HYPERPARAMETER TUNING-----#
        self.model = AdaBoostClassifier()
        #----------------------------------------------------------------#

        """
        ############### HYPERPARAMETER SEARCH  ###############
        #Create base model for search
        self.model_base = AdaBoostClassifier(**self.params)

        # Set Base Estimator Search Vector
        self.ada_base_estimator = [DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2),DecisionTreeClassifier(max_depth=3)]

        #Set number of estimators Search Vector
        self.ada_n_estimators = [int(x) for x in np.linspace(30,300,10)]

        #set Learning Rate Search Vector
        self.ada_learning_rate = [0.0001,0.001,0.01,0.1,1,10]

        #Create Hyperparameter Grid as Dictionary
        self.grid = {'base_estimator': self.ada_base_estimator,
                     'n_estimators': self.ada_n_estimators,
                     'criterion': self.rf_criterion,
                     'learning_rate': self.ada_learning_rate}

        #Create random search estimator
        self.model = GridSearchCV(estimator = self.model_base, param_grid = self.grid, cv = 5, verbose = 0, n_jobs = -1)
        ########## END HYPERPARAMETER SEARCH  #########
        """

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        #print(self.model.best_params_)

    def predict(self, X_test, y_test):
        raw_preds = self.model.predict_proba(X_test)
        self.score = self.model.score(X_test, y_test)
        preds = self.model.predict(X_test)
        return self.score, preds, raw_preds[:,1]

class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(X_train.shape[1], 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
    
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def train_neural_net(train_loader, epochs, learning_rate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = binaryClassification()
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for e in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()


        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
    return model

def test_neural_net(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    y_pred_list = []
    y_pred_list_raw = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_list_raw.append(y_test_pred.cpu().numpy()[0][0])
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    
    return np.array(y_pred_list), np.array(y_pred_list_raw)

def build_cnn_matrix(unprocessed_tweets):
    max_len = 0
    for e in unprocessed_tweets:
        if len(e) > max_len:
            max_len = len(e)

    unique_chars = set()
    for doc in unprocessed_tweets:
        for char in doc:
            unique_chars.add(char)

    char_hash = {}
    i = 0
    for c in sorted(unique_chars):
        char_hash[c] = i
        i += 1

    doc_list = []
    for doc in unprocessed_tweets:
        sample_mat = np.zeros([max_len, len(unique_chars)])
        for i, char in enumerate(doc):
            sample_mat[i,char_hash[char]] = 1
        doc_list.append(sample_mat)
    
    doc_mat = np.stack(doc_list, axis=0)

    return doc_mat

class binaryCNN(nn.Module):
    def __init__(self):
        super(binaryCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(X_train.shape[1], 10, kernel_size=10)
        self.conv2 = torch.nn.Conv1d(10, 20, kernel_size=20)

        self.layer_1 = nn.Linear(1360, 64) 
        self.layer_2 = nn.Linear(64, 20)
        self.layer_out = nn.Linear(20, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(20)
        
    def forward(self, inputs):
        x = self.relu(self.conv1(inputs))
        x = self.relu(self.conv2(x))

        x = x.view(x.shape[0], x.shape[1]*x.shape[2])
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

def train_cnn(train_loader, epochs, learning_rate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = binaryCNN()
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for e in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()


        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
    return model


if __name__ == "__main__":
    DATAFILE = "./Data/twitter_hate.csv"
    # D = Data(DATAFILE, preprocess=True)
    DE = False # boolean for checking if we're using doc embeddings or not. Just for convenience
    D = Data(DATAFILE, preprocess=False)
 
    no_emojis = []
    for doc in D.raw_tweets:
        no_emojis.append(D._fix_escaped_tokens(doc))

    mat = build_cnn_matrix(no_emojis)




    #### RUN WORD TFIDF #####
    # ngram_range = (2,2)
    # word_V = TFIDFWordVectorizer(D, ngram_range)
    # word_X = word_V.vectorize()
    # print('Word TFIDF shape: ', word_X.shape)

    ##### RUN CHAR TFIDF #####
    # ngram_range = (2,2)
    # char_V = TFIDFCharVectorizer(D, ngram_range)
    # char_X = char_V.vectorize()
    # print('Char TFIDF shape: ', char_X.shape)
    

    ##### RUN CHAR EMBEDDINGS #####
    # char_emb = CharEmbeddings(dataset=D, emb_path='./Data', emb_dim=300)
    # char_vecs = char_emb.vectorize()
    # print('Char embeddings shape: ', char_vecs.shape)

    #### RUN DOC EMBEDDINGS #####
    # DE = True
    # doc_emb = DocEmbeddings(D, emb_path="./Data", emb_dim=300)
    # doc_vecs = doc_emb.vectorize(success_rate=True)
    # print('Doc embeddings shape: ', doc_vecs.shape)


    #### Generate Labels #####
    if DE:
        labels = np.array(D.labels).astype(float)[doc_emb.vector_indices]
    else:
        labels = np.array(D.labels).astype(float)
    labels = np.where(labels < 0.5, 1, 0)
    print('Label shape: ', labels.shape)
   


    ##### Train test split #####
    X_train, X_test, y_train, y_test = train_test_split(mat, labels, test_size=0.33, random_state=42)

    # m = LogisticRegressionModel()
    # m = RandomForestModel()
    # m = SVMModel()
    # m = AdaBoostModel()

    # m.fit(X_train, y_train)
    # score, preds, raw_preds = m.predict(X_test, y_test)
    # print(score)
    # binary_metrics(y_test, raw_preds, preds)



    ##### Neural Network #####
    # if type(X_train) == csr_matrix:
    #     X_train = X_train.todense()
    #     X_test = X_test.todense()
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    # train_data = trainData(torch.FloatTensor(X_train), 
    #                     torch.FloatTensor(y_train))
    # test_data = testData(torch.FloatTensor(X_test))

    # train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    # test_loader = DataLoader(dataset=test_data, batch_size=1)

    # model = train_neural_net(train_loader, 10, 0.001)

    # preds, raw_preds = test_neural_net(model, test_loader)

    # binary_metrics(y_test, raw_preds, preds)


    ##### Convolutional Neural Network #####
    train_data = trainData(torch.FloatTensor(X_train), 
                        torch.FloatTensor(y_train))
    test_data = testData(torch.FloatTensor(X_test))

    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    model = train_cnn(train_loader, 10, 0.001)
    preds, raw_preds = test_neural_net(model, test_loader
    binary_metrics(y_test, raw_preds, preds)


    pass
