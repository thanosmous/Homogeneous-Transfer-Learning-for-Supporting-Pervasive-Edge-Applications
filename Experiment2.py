import time
import math
import numpy as nm
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from random import seed
from random import randint
from statistics import NormalDist
import statistics
from statsmodels.stats.weightstats import ztest as ztest
import scipy.stats
from numpy import sqrt, abs, round
from scipy.stats import norm
from scipy.stats import ks_2samp
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import random
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


random.seed(1)
start = time.time()

nm.random.seed(0)
'parameters'
p = 0.05
gamma = 0.8
beta = 0.9
nb=10
n = 200
k = 6

'datasets and data-batches'

def initialize(n, k, nb):  
    nm.random.seed(0)
    DS = list()
    DB = list()
    load = nm.empty(10)
    
    
    for i in range(10):
        load[i] = random.uniform(0,1)
    
    for i in range(10):
        y = nm.empty(n)
        d = list()
        a = 1
        b = 1
        c= (k/2) + 1
        for j in range(k):
            if(i % 3 == 0):
                nm.random.seed(0)
                x = nm.random.normal(a, a/10 + 0.1, n)
                a = a + 1
            elif(i % 3 == 1):
                nm.random.seed(0)
                x = nm.random.normal(b, b/10 + 0.1, n)
                if (b == k/2):
                    b = b + k/2 + 1
                else:
                    b = b + 1
            else:
                nm.random.seed(0)
                x = nm.random.normal(c, c/10 + 0.1, n)
                c = c + 1
                if (c == 4):
                    c = 6
            d.append(x)

        if(i % 3 == 0):
            for i in range(n):
                if d[0][i] > 1 and d[2][i] >3:
                    y[i] = 1
                else:
                    y[i] = 0
        elif(i % 3 == 1):
            for i in range(n):
                if d[0][i] > 1:
                    y[i] = 1
                else:
                    y[i] = 0
        else:
            for i in range(n):
                if d[0][i] > (k/2 +1):
                    y[i] = 1
                else:
                    y[i] = 0
        d.append(y)
        DS.append(d)
    
    for i in range(nb):
        y = nm.empty(n)
        g = list()
        a = 1
        b = 1
        c= (k/2) + 1
        for j in range(k):
            if(i % 3 == 0):
                nm.random.seed(0)
                x = nm.random.normal(a, a/10 + 0.1, n)
                a = a + 1
            elif(i % 3 == 1):
                nm.random.seed(0)
                x = nm.random.normal(b, b/10 + 0.1, n)
                if (b == k/2):
                    b = b + k/2 + 1
                else:
                    b = b + 1
            else:
                nm.random.seed(0)
                x = nm.random.normal(c, c/10 + 0.1, n)
                c = c + 1
                if (c == 4):
                    c = 6
            g.append(x)

        if(i % 3 == 0):
            for i in range(n):
                if g[0][i] > 1:
                    y[i] = 1
                else:
                    y[i] = 0
        elif(i % 3 == 1):
            for i in range(n):
                if g[0][i] > 1:
                    y[i] = 1
                else:
                    y[i] = 0
        else:
            for i in range(n):
                if g[0][i] > (k/2 + 1):
                    y[i] = 1
                else:
                    y[i] = 0
        g.append(y)
        DB.append(g)
    return DS,DB,load

    
    

DS,DB,load = initialize(n, k, nb)

lab = [1, 1, 1, 0, 1, 1, 1, 0, 0, 1]
def train(p, gamma, beta):
    'Train the models in the labeled datasets'
    'labeled or unlabeled datasets'
   
    m = list()
    l = nm.empty(10)
    
    for i in range(10):
        if(lab[i]==1):
            X = nm.empty((n,k))
            y = nm.empty(n)
            for j in range(n):
                for h in range(k):
                    X[j][h] = DS[i][h][j]
                y[j] = DS[i][k][j] 
                    
           
            #model = linear_model.LinearRegression()
            model = LogisticRegression(solver='liblinear', random_state=0)
            model.fit(X, y) 
            m.append(model.coef_)
            l[i] = model.intercept_
        else:
            m.append('null') 
            l[i] = 0
    nlba = 1
    lba = 0
    pr = 0
    re = 0
    f1 = 0
    "1o"
    model.intercept_ = l[1]
    model.coef_ = m[1]
    X = nm.empty((n,k))
    y = nm.empty(n)

    for j in range(n):
        for h in range(k):
            X[j][h] = DS[3][h][j]
        y[j] = DS[3][k][j] 
    model.predict_proba(X)
    ypred = model.predict(X)
    lba =lba + model.score(X, y)
    pr = pr + precision_score(ypred,y)
    re = re + recall_score(ypred,y)
    f1 = f1 + f1_score(ypred,y)
    "2o"
    model.intercept_ = l[5]
    model.coef_ = m[5]
    X = nm.empty((n,k))
    y = nm.empty(n)

    for j in range(n):
        for h in range(k):
            X[j][h] = DS[7][h][j]
        y[j] = DS[7][k][j] 
    model.predict_proba(X)
    ypred = model.predict(X)
    lba =lba + model.score(X, y)
    pr = pr + precision_score(ypred,y)
    re = re + recall_score(ypred,y)
    f1 = f1 + f1_score(ypred,y)
    "3o"
    model.intercept_ = l[6]
    model.coef_ = m[6]
    X = nm.empty((n,k))
    y = nm.empty(n)

    for j in range(n):
        for h in range(k):
            X[j][h] = DS[8][h][j]
        y[j] = DS[8][k][j] 
    model.predict_proba(X)
    ypred = model.predict(X)
    lba =lba + model.score(X, y)
    pr = pr + precision_score(ypred,y)
    re = re + recall_score(ypred,y)
    f1 = f1 + f1_score(ypred,y)
    
    
    
    return m, l,lba,pr,re,f1

m,l,lba,pr,re,f1 = train(p, gamma, beta)  





print('Accuracy',lba/3)    
print('Precision',pr/3)
print('Recall',re/3)
print('f1',f1/3)       
print('labeled', 3,'datasets')
print(time.time() - start)