# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 20:20:53 2019

@author: shahzaib laptopd
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m


# 0. Load in the data and split the descriptive and the target feature
df = pd.read_csv('vowel_train.csv')
X_train = df.iloc[:,2:].copy()
y = df['y'].copy()

df_test = pd.read_csv('vowel_test.csv')
X_test = df_test.iloc[:,2:].copy()
y_test = df_test['y'].copy()

X_train = X_train.values
# 1. Standardize the data
for i in range(10):
    var = X_train[:,i]
    m_v = np.sum(var)/len(var)
    s_v = np.sqrt(np.sum((var-(np.sum(var)/len(var)))**2)/len(var))
    X_train[:,i] = (var-m_v)/s_v       #Data is overwritten in "X" variable

X_train = pd.DataFrame(X_train)

X_test = X_test.values
for i in range(10):
    var = X_test[:,i]
    m_v = np.sum(var)/len(var)
    s_v = np.sqrt(np.sum((var-(np.sum(var)/len(var)))**2)/len(var))
    X_test[:,i] = (var-m_v)/s_v       #Data is overwritten in "X" variable

X_test = pd.DataFrame(X_test)


########################### LR Classifier ####################################

new_y = np.zeros((len(y), 11))
for i in range(len(y)):
    new_y[i,i%11] = 1
    
B = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T,X_train)), X_train.T), new_y)

y_pred = np.matmul(X_train,B)

new_y_test = np.zeros((len(y_test), 11))
for i in range(len(y_test)):
    new_y_test[i,i%11] = 1
y_test_pred = np.matmul(X_test,B)


for i in range(len(y_pred)):
    dat = y_pred[i,:]
    ab = np.where(dat == dat.max())
    for i in range(len(dat)):
        if i==ab[0][0]:
            dat[i] = 1
        else:
            dat[i] = 0
    y_pred[i,:] = dat
         
for i in range(len(y_test_pred)):
    dat = y_test_pred[i,:]
    ab = np.where(dat == dat.max())
    for i in range(len(dat)):
        if i==ab[0][0]:
            dat[i] = 1
        else:
            dat[i] = 0
    y_test_pred[i,:] = dat

prediction = y_pred
y_pred = np.zeros((len(prediction), 1))
for i in range(len(prediction)):
    row = prediction[i,:]
    for j in range (len(row)):
        if row[j] == 1:
            y_pred[i] = j
      
prediction_test = y_test_pred
y_test_pred = np.zeros((len(prediction_test), 1))
for i in range(len(prediction_test)):
    row = prediction_test[i,:]
    for j in range (len(row)):
        if row[j] == 1:
            y_test_pred[i] = j
                       
LR_accuracy_train = (y.values == y_pred).sum() / float(len(y.values))
LR_accuracy_test = (y_test.values == y_test_pred).sum() / float(len(y_test.values))

############################# LDA #######################################

# finding mean vector of each class
mu = np.mean(X_train,axis=0).values.reshape(10,1) # Mean vector mu --> Since the data has been standardized, the data means are zero 
mu_k = []
for i,orchid in enumerate(np.unique(df['y'])):
    mu_k.append(np.mean(X_train.where(df['y']==orchid),axis=0))
mu_k = np.array(mu_k).T
               
# finding covariance matrix
data_SW = []
Nc = []
for i,orchid in enumerate(np.unique(df['y'])):
    a = np.array(X_train.where(df['y']==orchid).dropna().values-mu_k[:,i].reshape(1,10))
    data_SW.append(np.dot(a.T,a))
    Nc.append(np.sum(df['y']==orchid))
Sw = np.sum(data_SW,axis=0)
Sb = np.dot(Nc*np.array(mu_k-mu),np.array(mu_k-mu).T)

cov_mat = Sw/(len(y)-11)

X_train = pd.DataFrame(X_train)

Beta = np.zeros((10,11))
for j in range(11):
    Beta[:,j] = np.matmul(np.linalg.inv(cov_mat),mu_k[:,j]-np.sum(np.delete(mu_k,j,1), axis = 1)/10)

lda_result_train = np.zeros((48,1))
for i in range(11):
    res = np.zeros((48,1))
    X_train_1 = X_train.where(df['y']==i+1).dropna()
    X_train_1 = X_train_1.T
    for k in range(48):
        res[k] = np.matmul((X_train_1.iloc[:,k].values-(np.sum(mu_k, axis = 1)/2)).T,Beta[:,i])
        if res[k]>-m.log10(48/528):
            res[k] = 1
        else:
            res[k] = 0
    
    if i == 0:
        lda_result_train = res
    else:
        lda_result_train = np.concatenate((lda_result_train, res), axis = 0)

lda_result_train = pd.DataFrame(lda_result_train)
LDA_accuracy_train = np.sum(lda_result_train.where(lda_result_train == 1).dropna())/528
print(LDA_accuracy_train)

 
lda_result_test = np.zeros((42,1))
for i in range(11):
    res = np.zeros((42,1))
    X_test_1 = X_test.where(df['y']==i+1).dropna()
    X_test_1 = X_test_1.T
    for k in range(42):
        res[k] = np.matmul((X_test_1.iloc[:,k].values-(np.sum(mu_k, axis = 1)/2)).T,Beta[:,i])
        if res[k]>-m.log10(48/528):
            res[k] = 1
        else:
            res[k] = 0
    
    if i == 0:
        lda_result_test = res
    else:
        lda_result_test = np.concatenate((lda_result_test, res), axis = 0)

lda_result_test = pd.DataFrame(lda_result_test)
LDA_accuracy_test = np.sum(lda_result_test.where(lda_result_test == 1).dropna())/462
print(LDA_accuracy_test)


############################# QDA #######################################

Beta_qda = np.zeros((10,11))
for j in range(11):
    Beta_qda[:,j] = np.matmul(np.linalg.inv(data_SW[i]),mu_k[:,j]-np.sum(np.delete(mu_k,j,1), axis = 1)/10)


qda_result_train = np.zeros((48,1))
for i in range(11):
    res = np.zeros((48,1))
    X_train_1 = X_train.where(df['y']==i+1).dropna()
    X_train_1 = X_train_1.T
    for k in range(48):
        res[k] = np.matmul((X_train_1.iloc[:,k].values-(np.sum(mu_k, axis = 1)/2)).T,Beta_qda[:,i])
        if res[k]>m.log10(42/462):
            res[k] = 1
        else:
            res[k] = 0
    
    if i == 0:
        qda_result_train = res
    else:
        qda_result_train = np.concatenate((qda_result_train, res), axis = 0)

qda_result_train = pd.DataFrame(qda_result_train)
QDA_accuracy_train = np.sum(qda_result_train.where(qda_result_train == 1).dropna())/528
print(QDA_accuracy_train)

 
qda_result_test = np.zeros((42,1))
for i in range(11):
    res = np.zeros((42,1))
    X_test_1 = X_test.where(df['y']==i+1).dropna()
    X_test_1 = X_test_1.T
    for k in range(42):
        res[k] = np.matmul((X_test_1.iloc[:,k].values-(np.sum(mu_k, axis = 1)/2)).T,Beta_qda[:,i])
        if res[k]>m.log10(42/462):
            res[k] = 1
        else:
            res[k] = 0
    
    if i == 0:
        qda_result_test = res
    else:
        qda_result_test = np.concatenate((qda_result_test, res), axis = 0)

qda_result_test = pd.DataFrame(qda_result_test)
QDA_accuracy_test = np.sum(qda_result_test.where(qda_result_test == 1).dropna())/462
print(QDA_accuracy_test)


###################### Visualizing Data in 2D #############################

# 4. Compute the Eigenvalues and Eigenvectors of SW^-1 SB
eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(Sw),Sb))
    

# 5. Select the two largest eigenvalues 
eigen_pairs = [[np.abs(eigval[i]),eigvec[:,i]] for i in range(len(eigval))]
eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0],reverse=True)
w = np.hstack((eigen_pairs[0][1][:,np.newaxis].real,eigen_pairs[1][1][:,np.newaxis].real)) # Select two largest

# 6. Transform the data with Y=X*w
Y = X_train.dot(w)

# Plot the data
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
for i in range(12):
    plt.scatter(Y[y==i+1][0], -Y[y==i+1][1])
