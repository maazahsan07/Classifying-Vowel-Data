# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 21:43:32 2019

@author: princ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m

def correlation_coefficient(T1, T2):
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result

# 0. Load in the data and split the descriptive and the target feature
df = pd.read_csv('vowel_train.csv')
X = df.iloc[:,2:].copy()
y = df['y'].copy()

df_test = pd.read_csv('vowel_test.csv')
X_test = df_test.iloc[:,2:].copy()
y_test = df_test['y'].copy()

X = X.values
# 1. Standardize the data
for i in range(10):
    var = X[:,i]
    m_v = np.sum(var)/len(var)
    s_v = np.sqrt(np.sum((var-(np.sum(var)/len(var)))**2)/len(var))
    X[:,i] = (var-m_v)/s_v       #Data is overwritten in "X" variable

X = pd.DataFrame(X)

X_test = X_test.values
for i in range(10):
    var = X_test[:,i]
    m_v = np.sum(var)/len(var)
    s_v = np.sqrt(np.sum((var-(np.sum(var)/len(var)))**2)/len(var))
    X_test[:,i] = (var-m_v)/s_v       #Data is overwritten in "X" variable

X_test = pd.DataFrame(X_test)

####################### LDA and Data Visualization for training #####################################

# 2. Compute the mean vector mu and the mean vector per class mu_k
mu = np.mean(X,axis=0).values.reshape(10,1) # Mean vector mu --> Since the data has been standardized, the data means are zero 
mu_k = []
for i,orchid in enumerate(np.unique(df['y'])):
    mu_k.append(np.mean(X.where(df['y']==orchid),axis=0))
mu_k = np.array(mu_k).T


# 3. Compute the Scatter within and Scatter between matrices
data_SW = []
Nc = []
for i,orchid in enumerate(np.unique(df['y'])):
    a = np.array(X.where(df['y']==orchid).dropna().values-mu_k[:,i].reshape(1,10))
    data_SW.append(np.dot(a.T,a))
    Nc.append(np.sum(df['y']==orchid))
Sw = np.sum(data_SW,axis=0)
Sb = np.dot(Nc*np.array(mu_k-mu),np.array(mu_k-mu).T)
   

# 4. Compute the Eigenvalues and Eigenvectors of SW^-1 SB
eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(Sw),Sb))
    

# 5. Select the two largest eigenvalues 
eigen_pairs = [[np.abs(eigval[i]),eigvec[:,i]] for i in range(len(eigval))]
eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0],reverse=True)
w = np.hstack((eigen_pairs[0][1][:,np.newaxis].real,eigen_pairs[1][1][:,np.newaxis].real)) # Select two largest

# 6. Transform the data with Y=X*w
Y = X.dot(w)

# Plot the data
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
for i in range(12):
    plt.scatter(Y[y==i+1][0], -Y[y==i+1][1])
    
######################## LDA and Data Visualization for test ##################################### 
#    
## 2. Compute the mean vector mu and the mean vector per class mu_k
#mu = np.mean(X_test,axis=0).values.reshape(10,1) # Mean vector mu --> Since the data has been standardized, the data means are zero 
#mu_k = []
#for i,orchid in enumerate(np.unique(df_test['y'])):
#    mu_k.append(np.mean(X_test.where(df_test['y']==orchid),axis=0))
#mu_k = np.array(mu_k).T
#
#
## 3. Compute the Scatter within and Scatter between matrices
#data_SW = []
#Nc = []
#for i,orchid in enumerate(np.unique(df_test['y'])):
#    a = np.array(X_test.where(df_test['y']==orchid).dropna().values-mu_k[:,i].reshape(1,10))
#    data_SW.append(np.dot(a.T,a))
#    Nc.append(np.sum(df_test['y']==orchid))
#SW = np.sum(data_SW,axis=0)
#SB = np.dot(Nc*np.array(mu_k-mu),np.array(mu_k-mu).T)
#   
#
## 4. Compute the Eigenvalues and Eigenvectors of SW^-1 SB
#eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(SW),SB))
#    
#
## 5. Select the two largest eigenvalues 
#eigen_pairs = [[np.abs(eigval[i]),eigvec[:,i]] for i in range(len(eigval))]
#eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0],reverse=True)
#w = np.hstack((eigen_pairs[0][1][:,np.newaxis].real,eigen_pairs[1][1][:,np.newaxis].real)) # Select two largest
#
## 6. Transform the data with Y=X*w
#Y_test_lda = X_test.dot(w)
#Y_test_lda = Y_test_lda.values

####################### Linear Regression #####################################

#X = X.T
#y_ohe = np.diag(np.ones(11))

new_y = np.zeros((len(y), 11))
for i in range(len(y)):
    new_y[i,i%11] = 1
    
B = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)), X.T), new_y)

Y_pred = np.matmul(X,B)

new_y_test = np.zeros((len(y_test), 11))
for i in range(len(y_test)):
    new_y_test[i,i%11] = 1
y_test_pred = np.matmul(X_test,B)


for i in range(len(Y_pred)):
    dat = Y_pred[i,:]
    ab = np.where(dat == dat.max())
    for i in range(len(dat)):
        if i==ab[0][0]:
            dat[i] = 1
        else:
            dat[i] = 0
    Y_pred[i,:] = dat
        

LR_train_err = 1-correlation_coefficient(Y_pred, new_y)
LR_test_err = 1-correlation_coefficient(y_test_pred, new_y_test)

####################### LDA #####################################

#B_lda = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)), X.T), Y)
#
#Y_pred_lda = np.matmul(X,B_lda)
#Y_pred_lda_test = np.matmul(X_test,B_lda)
#
#
#lda_train_err = correlation_coefficient(Y_pred_lda, Y.values)
#lda_test_err = correlation_coefficient(Y_pred_lda_test, Y_test_lda)

#B_lda = np.matmul(np.matmul(np.linalg.inv(np.matmul(Y.T,Y)), Y.T), new_y)
#
#Y_pred_lda = np.matmul(Y,B_lda)
#Y_pred_lda_test = np.matmul(Y_test_lda,B_lda)
#
#
#lda_train_err = correlation_coefficient(Y_pred_lda, new_y)
#lda_test_err = correlation_coefficient(Y_pred_lda_test, new_y_test)

#C = 48*Sw/len(X)
C = Sw/(len(y)-48)
Beta = np.zeros((10,11))
for i in range(11):
    Beta[:,i] = np.matmul(np.linalg.inv(C),mu_k[:,i]-np.sum(np.delete(mu_k,i,1), axis = 1))

res = np.zeros((len(y),1))
X = pd.DataFrame(X)
X = X.T
for i in range(len(y)):
    res[i] = np.matmul(Beta[:,i%11].T,X.iloc[:,i].values-(np.sum(mu_k, axis = 1)/2))
    

#m.log((48/len(y))/(len(y)-48/len(y)))

res_pred = np.zeros((len(y),1))
for i in range(len(y)):
    ma = np.matmul(np.matmul(X.iloc[:,i%11].values, Sw/(len(y)-48)), mu_k[:,i%11])
    n = np.matmul(np.matmul(mu_k[:,i%11].T, Sw/(len(y)-48)), mu_k[:,i%11])/2
    o = m.log(48/len(y))
    res_pred[i] = ma-n+o

for i in range(len(res)):
    if res[i]>11:
        res[i] = 11
    else:
        res[i] = round(res[i,0])

####################### QDA #####################################

