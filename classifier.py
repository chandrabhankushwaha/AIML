import pandas as pd
import numpy as np
import sys
from scipy.stats import randint
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import metrics
from tqdm import tqdm
from numpy import save
from numpy import load
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
np.set_printoptions(threshold=sys.maxsize)

'''
Responsible for:
    encoding data
    training the model
    classifying the types of errors for test data
'''


# Maximum possible input (encoding) length
MAX_ENC_LEN= 15

# Set of characters considered for encoding
char_set = list(" .abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
char2int = { char_set[x]:x for x in range(len(char_set)) }
int2char = { char2int[x]:x for x in char_set }

# Encoding string
def encode_string(ip_text):
    main=[]
    for t,char in enumerate(ip_text):
        temp=np.zeros((1, len(char_set)),dtype='float32')
        temp[0, char2int[char]]=1
        main = np.append(main, temp)

    main = np.append(main, np.zeros((1, (MAX_ENC_LEN-len(ip_text))*len(char_set))))

    return main

# Encoding training data
def enc_ip(X_train):
    X_enc=np.empty((0,7680))
    for i in range(len(X_train)):
        temp =[]
        for col in X_train.columns:
            temp = np.append(temp, encode_string(str(X_train.iloc[i][col])))

        X_enc = np.append(X_enc, temp)
    X_enc = np.reshape(X_enc, (len(X_train), 7680))
    return X_enc

# Train model
def train_model():
    df = pd.read_excel('input_data/TrainData4.xlsx')
    X_train = df.iloc[:,1:9]
    Y_train = df.iloc[:,-1]
    X_enc = load('input_data/X_Train4.npy')
    X_train, X_test, y_train, y_test = train_test_split(X_enc, Y_train, test_size=0.2, random_state=0)
    model = QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train)
    return model

# Testing model
def test():
    df2=pd.read_excel('input_data/TestData4.xlsx')
    x_testdata3 = df2.iloc[:,1:9]
    y_testdata3 = df2.iloc[:,-1]
    x_testdata3 = load('input_data/X_Test4.npy')
    y_pred = model.predict(x_testdata3)
    print('Accuracy=', accuracy_score(y_testdata3, y_pred))

# Classifying type of error
def classify_error(input):
    input=input[["PmtInfId","Dbtr.Nm","Dbtr.ActNo","Dbtr.DbtrAgt","Dbtr.BIC","Cdtr.Nm","Cdtr.ActNo","Cdtr.CdtrAgt","Cdtr.BIC"]]
    X_test = input.iloc[:,0:8]
    y_pred = model.predict(enc_ip(X_test))
    
    return y_pred

model=train_model()
