from preprocessing import Data
from vectorize import TFIDFCharVectorizer
from metrics import binary_metrics

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler    

from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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


class RandomForestModel():
    def __init__(self):
        self.model = RandomForestClassifier()
        self.score = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test, y_test):
        self.model.predict_proba(X_test)
        self.score = self.model.score(X_test, y_test)
        preds = self.model.predict(X_test)
        return self.score, preds

class SVMModel():
    def __init__(self):
        self.params = {
            "C": 1.0,
            "kernel": 'rbf',
            "gamma": 'scale',
            "probability": True
        }
        self.model = svm.SVC(**self.params)
        self.score = None
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test, y_test):
        self.model.predict_proba(X_test)
        self.score = self.model.score(X_test, y_test)
        preds = self.model.predict(X_test)
        return self.score, preds

class AdaBoostModel():
    def __init__(self):
        self.params = {
            "n_estimators": 50,
            "learning_rate": 1
        }
        self.model = AdaBoostClassifier(**self.params)
        self.score = None
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test, y_test):
        self.model.predict_proba(X_test)
        self.score = self.model.score(X_test, y_test)
        preds = self.model.predict(X_test)
        return self.score, preds

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.linear1 = torch.nn.Linear(X_train.shape[1], 64)
        self.linear2 = torch.nn.Linear(64, 64)
        self.linear3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


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


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


def train_neural_net(X_train, X_test, y_train, epochs, batch_size, learning_rate):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    print(X_train)
    
    train_data = trainData(torch.FloatTensor(X_train), 
                       torch.FloatTensor(y_train))

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = NeuralNet()
    model.to(device)
    print(model)
    criterion = torch.nn.BCEWithLogitsLoss()
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


def test_neural_net(model, y_test): 
    test_data = testData(torch.FloatTensor(X_test))
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    y_pred_list = []
    y_pred_list_round = []

    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_list.append(float(y_test_pred.cpu().numpy()))
            y_test_pred_round = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred_round)
            y_pred_list_round.append(y_pred_tag.cpu().numpy())

    y_pred_list_round = [a.squeeze().tolist() for a in y_pred_list_round]
    
    return y_pred_list, y_pred_list_round


if __name__ == "__main__":
    DATAFILE = "./Data/twitter_hate.csv"
    D = Data(DATAFILE, preprocess=False)

    ngram_range = (2,2)
    char_V = TFIDFCharVectorizer(D, ngram_range)
    char_X = char_V.vectorize()
    print('Char TFIDF shape: ', char_X.shape)

    labels = np.array(D.labels).astype(float)
    labels = np.where(labels < 0.5, 1, 0)
    print('Label shape: ', labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(char_X, labels, test_size=0.33, random_state=42)

    # m = LogisticRegressionModel()
    # m = RandomForestModel()
    # m = SVMModel()
    m = AdaBoostModel()
    
    #model = train_neural_net(X_train, X_test, y_train, epochs=10, batch_size=64, learning_rate=0.001)
    #y_pred, y_pred_rounded = test_neural_net(model, y_test)
    #binary_metrics(y_test, y_pred, y_pred_rounded)


    m.fit(X_train, y_train)
    score, preds = m.predict(X_test, y_test)
    print('Score: ', score)
    print(confusion_matrix(preds,y_test))

    pass
