
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

random.seed(1)

def fuzzy1(cost,sim):
    'FUZZY'
    
    x_cost = nm.arange(0,1,0.001)
    x_sim = nm.arange(0,1,0.001)
    y_DLC = nm.arange(0,1,0.001)
    
    
    cost_low = mf.trapmf(x_cost, [-0.4, -0.2, 0.2, 0.4])
    cost_medium = mf.trapmf(x_cost, [0.2, 0.4, 0.6, 0.8])
    cost_high= mf.trapmf(x_cost, [0.6, 0.7, 0.8, 200])
    
    sim_low = mf.trapmf(x_sim, [-0.4, -0.2, 0.2, 0.41])
    sim_medium = mf.trapmf(x_sim, [0.2, 0.4, 0.6, 0.8])
    sim_high= mf.trapmf(x_sim, [0.6, 0.7, 0.8, 200])
    
    cost_fit_low = fuzz.interp_membership(x_cost , cost_low, cost)
    cost_fit_medium = fuzz.interp_membership(x_cost , cost_medium, cost)
    cost_fit_high = fuzz.interp_membership(x_cost ,cost_high, cost)
    
    sim_fit_low = fuzz.interp_membership(x_sim , sim_low, sim)
    sim_fit_medium = fuzz.interp_membership(x_sim , sim_medium, sim)
    sim_fit_high = fuzz.interp_membership(x_sim ,sim_high, sim)
    
    DLC_human = mf.trapmf(y_DLC, [-0.3, 0 ,0.2 , 0.25])
    DLC_peer = mf.trapmf(y_DLC, [0.2, 0.23 ,0.64 , 0.7])
    DLC_local = mf.trapmf(y_DLC, [0.64, 0.67 ,1 , 200])
    
    
    
    rule1 = nm.fmin(nm.fmin(cost_fit_low, sim_fit_low), DLC_human)
    rule2 = nm.fmin(nm.fmin(cost_fit_medium, sim_fit_low), DLC_peer)
    rule3 = nm.fmin(nm.fmin(cost_fit_high, sim_fit_low), DLC_peer)
    
    rule4 = nm.fmin(nm.fmin(cost_fit_low, sim_fit_medium), DLC_human)
    rule5 = nm.fmin(nm.fmin(cost_fit_medium, sim_fit_medium), DLC_peer)
    rule6 = nm.fmin(nm.fmin(cost_fit_high, sim_fit_medium), DLC_peer)
    
    rule7 = nm.fmin(nm.fmin(cost_fit_low, sim_fit_high), DLC_local)
    rule8 = nm.fmin(nm.fmin(cost_fit_medium, sim_fit_high), DLC_local)
    rule9 = nm.fmin(nm.fmin(cost_fit_high, sim_fit_high), DLC_local)
    
    out_human = nm.fmax(rule1 , rule4)
    out_peer = nm.fmax(nm.fmax(nm.fmin(rule2 , rule3),rule5),rule6)
    out_local = nm.fmax(nm.fmax(rule7 , rule8), rule9)
    
    out = nm.fmax(nm.fmax(out_human, out_peer), out_local)
    try:
        r= nm.empty(3)
        defuzzified  = fuzz.defuzz(y_DLC, out, 'centroid')   
        r[0] = fuzz.interp_membership(y_DLC, out_human, defuzzified)
        r[1] = fuzz.interp_membership(y_DLC, out_peer, defuzzified)
        r[2] = fuzz.interp_membership(y_DLC, out_local, defuzzified)
        if r[0] == max(r):
            return 'human'
        elif r[1] == max(r):
            return 'peer'
        else:
            return'local'
    except:
        'print(CAUGHT)'
        return 'peer'

def fuzzy2(load,sim):
    'FUZZY'
    
    x_load = nm.arange(0,1,0.001)
    x_load = nm.arange(0,1,0.001)
    x_sim = nm.arange(0,1,0.001)
    y_DAT = nm.arange(0,1,0.001)
    
    
    load_low = mf.trapmf(x_load, [-0.4, -0.2, 0.2, 0.4])
    load_medium = mf.trapmf(x_load, [0.2, 0.4, 0.6, 0.8])
    load_high= mf.trapmf(x_load, [0.6, 0.7, 0.8, 200])
    
    sim_low = mf.trapmf(x_sim, [-0.4, -0.2, 0.2, 0.41])
    sim_medium = mf.trapmf(x_sim, [0.2, 0.4, 0.6, 0.8])
    sim_high= mf.trapmf(x_sim, [0.6, 0.7, 0.8, 200])
    
    load_fit_low = fuzz.interp_membership(x_load , load_low, load)
    load_fit_medium = fuzz.interp_membership(x_load , load_medium, load)
    load_fit_high = fuzz.interp_membership(x_load ,load_high, load)
    
    sim_fit_low = fuzz.interp_membership(x_sim , sim_low, sim)
    sim_fit_medium = fuzz.interp_membership(x_sim , sim_medium, sim)
    sim_fit_high = fuzz.interp_membership(x_sim ,sim_high, sim)
    
    DAT1 = mf.trapmf(y_DAT, [-0.3, 0 ,0.2 , 0.25])
    DAT2 = mf.trapmf(y_DAT, [0.2, 0.23 ,0.64 , 0.7])
    DAT3 = mf.trapmf(y_DAT, [0.64, 0.67 ,1 , 200])
    
    
    
    rule1 = nm.fmin(nm.fmin(load_fit_low, sim_fit_low), DAT1)
    rule2 = nm.fmin(nm.fmin(load_fit_medium, sim_fit_low), DAT1)
    rule3 = nm.fmin(nm.fmin(load_fit_high, sim_fit_low), DAT3)
    
    rule4 = nm.fmin(nm.fmin(load_fit_low, sim_fit_medium), DAT1)
    rule5 = nm.fmin(nm.fmin(load_fit_medium, sim_fit_medium), DAT1)
    rule6 = nm.fmin(nm.fmin(load_fit_high, sim_fit_medium), DAT3)
    
    rule7 = nm.fmin(nm.fmin(load_fit_low, sim_fit_high), DAT2)
    rule8 = nm.fmin(nm.fmin(load_fit_medium, sim_fit_high), DAT2)
    rule9 = nm.fmin(nm.fmin(load_fit_high, sim_fit_high), DAT2)
    
    out1 = nm.fmax(nm.fmax(nm.fmax(rule1 , rule2),rule4),rule5)
    out2 = nm.fmax(nm.fmax(rule7 , rule8), rule9)
    out3 = nm.fmax(rule3 , rule6)
    
    out = nm.fmax(nm.fmax(out1, out2), out3)
    try:
        r= nm.empty(3)
        defuzzified  = fuzz.defuzz(y_DAT, out, 'centroid')   
        r[0] = fuzz.interp_membership(y_DAT, out1, defuzzified)
        r[1] = fuzz.interp_membership(y_DAT, out2, defuzzified)
        r[2] = fuzz.interp_membership(y_DAT, out3, defuzzified)
        if r[0] == max(r):
            return 'save/train'
        elif r[1] == max(r):
            return 'save/no train'
        else:
            return'no save/no train'
    except:
        'print(caught)'
        return 'what now'




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
gamma = 0.6
beta = 0.6

n = 200
'mono 4, 5, 6 to k '
k = 6
nb = 100
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
        c = 1
        for j in range(k):
            if(i % 3 == 0):
                nm.random.seed(0)
                x = nm.random.normal(a, a/10 + 0.1, n)
                a = a + 1
            elif(i % 3 == 1):
                nm.random.seed(0)
                x = nm.random.normal(b, b/10 + 0.1, n)
                if (b == 4):
                    b = 7
                else:
                    b = b + 1
            else:
                nm.random.seed(0)
                x = nm.random.normal(c, c/10 + 0.1, n)
                c = c + 1
                if (c == 5):
                    c = 8
                else:
                    c = c + 1
            d.append(x)

        if(i % 3 == 0):
            for i in range(n):
                if d[0][i] > 1 and d[1][i] >2 and d[5][i] >6:
                    y[i] = 1
                else:
                    y[i] = 0
        elif(i % 3 == 1):
            for i in range(n):
                if d[3][i] > 4:
                    y[i] = 1
                else:
                    y[i] = 0
        else:
            for i in range(n):
                if d[5][i] > 8:
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
                if (b == 4):
                    b = 7
                else:
                    b = b + 1
            else:
                nm.random.seed(0)
                x = nm.random.normal(c, c/10 + 0.1, n)
                if (c == 5):
                    c = 8
                else:
                    c = c + 1
            g.append(x)

        if(i % 3 == 0):
            for i in range(n):
                if g[0][i] > 1 and g[1][i] >2 and g[5][i] >6:
                    y[i] = 1
                else:
                    y[i] = 0
        elif(i % 3 == 1):
            for i in range(n):
                if g[3][i] > 4:
                    y[i] = 1
                else:
                    y[i] = 0
        else:
            for i in range(n):
                if g[5][i] > 8:
                    y[i] = 1
                else:
                    y[i] = 0
        g.append(y)
        DB.append(g)
    return DS,DB,load

    
    

DS,DB,load = initialize(n, k, nb)

'fTest'


lab = [1, 1, 1, 0, 1, 1, 1, 0, 0, 1]
def train(p, gamma, beta,nb):
    'Train the models in the labeled datasets'
    'labeled or unlabeled datasets'
    lba = 0
    nlba = 0
    lba1 = 0
    nlba1 = 0
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
    
                        if(twoSampZ(x1, x2, 0, s1, s2, n, n) < p or f_test(DS[i][h], DS[j][h]) < p):
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
                model.predict(X)
                print(i,max_index,model.score(X, DS[i][k]))
                
                lba1 =lba1 + model.score(X, DS[i][k])
                nlba1 = nlba1 +1
                
    print('accuracy datasets',lba1/nlba1)
    
    
    for i in range(10):
        if(lab[i]==0):
            lab[i] = 1
            X = nm.empty((n,k))
            y = nm.empty(n)
            for j in range(n):
                for h in range(k):
                    X[j][h] = DS[i][h][j]
                y[j] = DS[i][k][j] 
                    
            
            model = LogisticRegression(solver='liblinear', random_state=0)
            model.fit(X, y)
            'print( model.score(X, y))'
            m[i] = model.coef_
            l[i] = model.intercept_
            '''regr = linear_model.LinearRegression()
            regr.fit(X, y) 
            m[i] = regr.coef_
            l[i] = regr.intercept_'''
        
            
    
    'LABEL BATCHES'
    
    for i in range(nb):
        ld = random.uniform(0.01,0.03)
        Sl1 = Sl1 + ld
        cost = random.uniform(0.1,0.8)
        Sc1 = Sc1 + cost
        while(1):
            node = random.randint(0,9)
            if(l[node] != 0):
                break;
            
            node = random.randint(0,9)
        sum2 = 0
        for h in range(k):

            if(ks_2samp(DS[node][h], DB[i][h])[1] < beta):
                sum2 = sum2 + 0
            else:
                sum2 = sum2 + 1
        if sum2 == k:
            sim = 0.99
        else:
            sim = sum2/k
            
        result1 = fuzzy1(cost,sim)
        result2 = fuzzy2(load[node],sim)
        'print(sum2/k,result2)'
        if result2 == 'save/no train':
            Sl2 = Sl2 + ld
        'if(sum2/4 >= beta):'
        if result1 == 'local':
            Sc2 = Sc2 + cost
            X = nm.empty((n, k))
            for j in range(n):
                for h in range(k):
                    X[j][h] = DB[i][h][j]
            model.intercept_ = l[node]
            model.coef_ = m[node]
            model.predict_proba(X)
            model.predict(X)
            'print(i,node,model.score(X, DB[i][k]))'
            lba =lba + model.score(X, DB[i][k])
            nlba = nlba +1
            
        elif result1 =='peer':  
            Sc2 = Sc2 + cost
            sim = nm.empty(10)
            dissim = nm.empty(10)
            for j in range(10):
                sum1 = 0
                if (j != node):
                   
                    for h in range(k):
                        'ztest(DS[i][k], DS[j][k], value=0)'
                        x1 = nm.mean(DS[j][h])
                        x2 = nm.mean(DB[i][h])
                        s1 = nm.std(DS[j][h])
                        s2 = nm.std(DB[i][h])
    
                        if(twoSampZ(x1, x2, 0, s1, s2, n, n) < p or f_test(DS[j][h], DB[i][h]) < p):
                            sum1 = sum1 + 1
                    'print(i,j,sum1)'
                    if(sum1/k >= 1-gamma):
                        dissim[j] = 1
                    else:
                        dissim[j] = 0
            
                
            dissim[node] = 0
            for j in range(10):
                if (dissim[j] == 0 & j != node):
                    sum2 = 0
                    for h in range(k):
    
                        if(ks_2samp(DS[j][h], DB[i][h])[1] < beta):
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
                    X[j][h] = DB[i][h][j]
                
    
            max_val = 0
            max_index = 0
            for h in range(10):
                if (h != node and sim[h] >= gamma):
                    max_val = sim[h]
                    max_index = h
                    break;
                    'print(MAX,max_index)'
            'print(i, max_index,model.score(X, DB[i][k]))'
            if(max_val >= gamma):
                model.intercept_ = l[max_index]
                model.coef_ = m[max_index]
                model.predict_proba(X)
                model.predict(X)
                'print(i,max_index,model.score(X, DB[i][k])) '
                lba =lba + model.score(X, DB[i][k])
                nlba = nlba + 1
        else:
            Sc2 = Sc2 + 0
    print('ls',round(Sl2/Sl1,2) * 100)
    print('cs',round(Sc2/Sc1,2) * 100)
    print('accuracy',round(lba/nlba,2)*100)
    return m
    ',nlba,lba/nlba,Sl1,Sl2,Sc1,Sc2'
         
'initialize(n, k, ba)'  
m = train(p, gamma, beta,nb)


'''
DS,DB = initialize(n, k, 50)
print(train(p, gamma, beta,50))
'''
