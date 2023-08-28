import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp
import math
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")
data1 = pd.read_csv(r'400-600.csv')
data2 = pd.read_csv(r'400-1100.csv')

data1 = data1.drop('Date/Time',axis=1)
data2 = data2.drop('Date/Time',axis=1)

#4 datasets for buildings A,B,C,D

data3 = data1.iloc[:864,:]
data4 = data1.iloc[864:,:]
data5 = data2.iloc[:864,:]
data6 = data2.iloc[864:,:]

sc_X = StandardScaler()
sc_y = StandardScaler()


#model for building A

X1 = data3[['CO2','Humidity','Temperature']]
Y1 = data3[['Occupancy']] 

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1 ,
                                   random_state=104, 
                                   test_size=0.30, 
                                   shuffle=True)

X1_train = sc_X.fit_transform(X1_train)
Y1_train = sc_y.fit_transform(Y1_train)
X1_test = sc_X.fit_transform(X1_test)
'Y1_test = sc_y.fit_transform(Y1_test)'
regressor1 = SVR(kernel = 'rbf')
regressor1.fit(X1_train, Y1_train)
Y1_pred = regressor1.predict(X1_test)
Y1_pred = Y1_pred.reshape(-1,1)
Y1_pred = sc_y.inverse_transform(Y1_pred)


#model for building C
X2 = data5[['CO2','Humidity','Temperature']]
Y2 = data5[['Occupancy']] 

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2 ,
                                   random_state=104, 
                                   test_size=0.30, 
                                   shuffle=True)


X2_train = sc_X.fit_transform(X2_train)
Y2_train = sc_y.fit_transform(Y2_train)
X2_test = sc_X.fit_transform(X2_test)
'Y1_test = sc_y.fit_transform(Y2_test)'
regressor2 = SVR(kernel = 'rbf')
regressor2.fit(X2_train, Y2_train)
Y2_pred = regressor2.predict(X2_test)
Y2_pred = Y2_pred.reshape(-1,1)
Y2_pred = sc_y.inverse_transform(Y2_pred)

#model from data A,C

data7 = data5.iloc[:200]
frames = [data3,data7]
result = pd.concat(frames)

X2 = result[['CO2','Humidity','Temperature']]
Y2 = result[['Occupancy']] 

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2 ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)


X2_train = sc_X.fit_transform(X2_train)
Y2_train = sc_y.fit_transform(Y2_train)
X2_test = sc_X.fit_transform(X2_test)
'Y1_test = sc_y.fit_transform(Y1_test)'
regressor3 = SVR(kernel = 'rbf')
regressor3.fit(X2_train, Y2_train)
Y2_pred = regressor3.predict(X2_test)
Y2_pred = Y2_pred.reshape(-1,1)
Y2_pred = sc_y.inverse_transform(Y2_pred)

print("model from A,C applied to predict D")

X2 = data6[['CO2','Humidity','Temperature']]
Y2 = data6[['Occupancy']] 

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2 ,
                                   random_state=104, 
                                   test_size=0.30, 
                                   shuffle=True)




Y2_pred = regressor3.predict(X2)
Y2_pred = Y2_pred.reshape(-1,1)
Y2_pred = sc_y.inverse_transform(Y2_pred)

testScore = np.sqrt(mean_squared_error(Y2,Y2_pred))
print('RMSE',testScore)    
print('MAE', mean_absolute_error(Y2,Y2_pred))
print('R2', mean_absolute_error(Y2,Y2_pred))

print("model from C applied to predict D")
Y2_pred = regressor2.predict(X2)
Y2_pred = Y2_pred.reshape(-1,1)
Y2_pred = sc_y.inverse_transform(Y2_pred)

testScore = np.sqrt(mean_squared_error(Y2,Y2_pred))
print('RMSE',testScore)    
print('MAE', mean_absolute_error(Y2,Y2_pred))
print('R2', mean_absolute_error(Y2,Y2_pred)) 

print("model from A applied to predict D")
Y2_pred = regressor1.predict(X2)
Y2_pred = Y2_pred.reshape(-1,1)
Y2_pred = sc_y.inverse_transform(Y2_pred)

testScore = np.sqrt(mean_squared_error(Y2,Y2_pred))
print('RMSE',testScore)    
print('MAE', mean_absolute_error(Y2,Y2_pred))
print('R2', mean_absolute_error(Y2,Y2_pred))

#model from data C,A

data8 = data3.iloc[:200]
frames = [data5,data8]
result = pd.concat(frames)

X2 = result[['CO2','Humidity','Temperature']]
Y2 = result[['Occupancy']] 

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2 ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)


X2_train = sc_X.fit_transform(X2_train)
Y2_train = sc_y.fit_transform(Y2_train)
X2_test = sc_X.fit_transform(X2_test)
'Y1_test = sc_y.fit_transform(Y1_test)'
regressor4 = SVR(kernel = 'rbf')
regressor4.fit(X2_train, Y2_train)
Y2_pred = regressor4.predict(X2_test)
Y2_pred = Y2_pred.reshape(-1,1)
Y2_pred = sc_y.inverse_transform(Y2_pred)

print("model from C,A applied to predict B")

X2 = data4[['CO2','Humidity','Temperature']]
Y2 = data4[['Occupancy']] 

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2 ,
                                   random_state=104, 
                                   test_size=0.30, 
                                   shuffle=True)




Y2_pred = regressor4.predict(X2)
Y2_pred = Y2_pred.reshape(-1,1)
Y2_pred = sc_y.inverse_transform(Y2_pred)

testScore = np.sqrt(mean_squared_error(Y2,Y2_pred))
print('RMSE',testScore)    
print('MAE', mean_absolute_error(Y2,Y2_pred))
print('R2', mean_absolute_error(Y2,Y2_pred))   

print("model from C applied to predict B")
Y2_pred = regressor2.predict(X2)
Y2_pred = Y2_pred.reshape(-1,1)
Y2_pred = sc_y.inverse_transform(Y2_pred)

testScore = np.sqrt(mean_squared_error(Y2,Y2_pred))
print('RMSE',testScore)    
print('MAE', mean_absolute_error(Y2,Y2_pred))
print('R2', mean_absolute_error(Y2,Y2_pred))

print("model from A applied to predict B")
Y2_pred = regressor1.predict(X2)
Y2_pred = Y2_pred.reshape(-1,1)
Y2_pred = sc_y.inverse_transform(Y2_pred)

testScore = np.sqrt(mean_squared_error(Y2,Y2_pred))
print('RMSE',testScore)    
print('MAE', mean_absolute_error(Y2,Y2_pred))
print('R2', mean_absolute_error(Y2,Y2_pred))

#SIMILARITY

def f_test(group1, group2):
    f = nm.var(group1, ddof=1)/nm.var(group2, ddof=1)
    nun = group1.size-1
    dun = group2.size-1
    p_value = 1-scipy.stats.f.cdf(f, nun, dun)
    return p_value


def twoSampZ(X1, X2, mudiff, sd1, sd2, n1, n2):
    x = (sd1**2/n1) + (sd2**2/n2)
    pooledSE = math.sqrt(x)
    z = ((X1 - X2) - mudiff)/pooledSE
    pval = 2*(norm.sf(abs(z)))
    return round(pval, 4)

X1 = data4[['CO2','Humidity','Temperature']]
X2 = data5[['CO2','Humidity','Temperature']]


simd = 0

n = 864
p = 0.05
gamma = 0.3
beta = 0.3
k = 3


sum2 = 0

v = X1["CO2"]  
print(ks_2samp(X1["CO2"].to_numpy(), X2["CO2"].to_numpy()))       
print(ks_2samp(X1["Humidity"].to_numpy(), X2["Humidity"].to_numpy()))
print(ks_2samp(X1["Temperature"].to_numpy(), X2["Temperature"].to_numpy()))        
if(ks_2samp(X1["CO2"].to_numpy(), X2["CO2"].to_numpy())[0] < beta):
    sum2 = sum2 + 0
else:
    sum2 = sum2 + 1

if(ks_2samp(X1["Humidity"].to_numpy(), X2["Humidity"].to_numpy())[0] < beta):
    sum2 = sum2 + 0
else:
    sum2 = sum2 + 1

if(ks_2samp(X1["Temperature"].to_numpy(), X2["Temperature"].to_numpy())[0] < beta):
    sum2 = sum2 + 0
else:
    sum2 = sum2 + 1
    
if(sum2/k < beta):
    sim = 0
else:
    sim = sum2/k
    
print('sim:',sim)