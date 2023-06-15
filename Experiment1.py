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
def f_test(group1, group2):
    f = nm.var(group1, ddof=1)/nm.var(group2, ddof=1)
    nun = group1.size-1
    dun = group2.size-1
    p_value = 1-scipy.stats.f.cdf(f, nun, dun)
    return p_value


def twoSampZ(X1, X2, mudiff, sd1, sd2, n1, n2):
    x = (sd1**2/n1) + (sd2**2/n2)
    pooledSE = sqrt(x)
    z = ((X1 - X2) - mudiff)/pooledSE
    pval = 2*(norm.sf(abs(z)))
    return round(pval, 4)

nm.random.seed(0)
'parameters'
p = 0.05
gamma = 0.8
beta = 0.8

n = 200
k = 6
nb = 10
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
    lba = 0
    nlba = 0
    pr = 0
    re = 0
    f1 = 0
    m = list()
    Sl1 = 0
    Sc1 = 0
    Sl2 = 0
    Sc2 = 0
    l = nm.empty(10)
    for i in range(10):
        if(lab[i]==1):
            X = nm.empty((n,k))
            y = nm.empty(n)
            for j in range(n):
                for h in range(k):
                    X[j][h] = DS[i][h][j]
                y[j] = DS[i][k][j] 
                    
            
            model = LogisticRegression(solver='liblinear', random_state=0)
            model.fit(X, y)
            'print( model.score(X, y))'
            m.append(model.coef_)
            l[i] = model.intercept_
            '''regr = linear_model.LinearRegression()
            regr.fit(X, y) 
            m[i] = regr.coef_
            l[i] = regr.intercept_'''
        else:
            m.append('null') 
            l[i] = 0
    


    'LABEL DATASETS'
    simd = nm.empty(10)
    for i in range(10):
        if(lab[i] == 0):
            sim = nm.empty(10)
            dissim = nm.empty(10)
            for j in range(10):
                sum1 = 0
                if (j != i):
                   
                    for h in range(k):
                        'ztest(DS[i][k], DS[j][k], value=0)'
                        x1 = nm.mean(DS[i][h])
                        x2 = nm.mean(DS[j][h])
                        s1 = nm.std(DS[i][h])
                        s2 = nm.std(DS[j][h])
    
                        if(twoSampZ(x1, x2, 0, s1, s2, n, n) < p or f_test(DS[i][k], DS[j][k]) < p):
                            sum1 = sum1 + 1
                    'print(i,j,sum1)'
                    if(sum1/k >= 1-gamma):
                        dissim[j] = 1
                    else:
                        dissim[j] = 0
            dissim[i] = 0
            for j in range(10):
                if (dissim[j] == 0 & j != i):
                    sum2 = 0
                    for h in range(k):
                        'print(ks_2samp(DS[i][h], DS[j][h])[1])'
                        if(ks_2samp(DS[i][h], DS[j][h])[1] < beta):
                            sum2 = sum2 + 0
                        else:
                            sum2 = sum2 + 1
    
                    if(sum2/k < beta):
                        sim[j] = 0
                    else:
                        sim[j] = sum2/k
                else:
                    sim[j] = 0
    
            X = nm.empty((n, k))
            for j in range(n):
                for h in range(k):
                    X[j][h] = DS[i][h][j]
                
    
            max_val = 0
            max_index = 0
            for h in range(10):
                'gamma anti gia sim[h]'
                if (h != i and sim[h] >= gamma):
                    max_val = sim[h]
                    max_index = h
                    simd[i] = h
                    break;
            'print(i, max_index)'
            if(max_val >= gamma):
                model.intercept_ = l[max_index]
                model.coef_ = m[max_index]
                model.predict_proba(X)
                ypred = model.predict(X)
                'print(i,max_index,model.score(X, DS[i][k]))'
                lba =lba + model.score(X, DS[i][k])
                pr = pr + precision_score(ypred,DS[i][k])
                re = re + recall_score(ypred,DS[i][k])
                f1 = f1 + f1_score(ypred,DS[i][k])
                nlba = nlba + 1
    print(lba/nlba)
    print('Precision',pr/nlba)
    print('Recall',re/nlba)
    print('f1',f1/nlba)             
    print('labeled', nlba,'datasets')
    #print(time.time() - start)
train(p, gamma, beta)    