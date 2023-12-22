# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 03:29:06 2019

@author: princ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation as LDA


df = pd.read_csv('vowel_train.csv')
X = df.iloc[:,2:]
y = df['y']

df_test = pd.read_csv('vowel_test.csv')
X_test = df_test.iloc[:,2:]
y_test = df_test['y']

#X_norm = (X - X.min())/(X.max() - X.min())

#X = X.values
## 1. Standardize the data
#for i in range(10):
#    var = X[:,i]
#    m_v = np.sum(var)/len(var)
#    s_v = np.sqrt(np.sum((var-(np.sum(var)/len(var)))**2)/len(var))
#    X[:,i] = (var-m_v)/s_v       #Data is overwritten in "X" variable
#
#X = pd.DataFrame(X)
#
#X_test = X_test.values
#for i in range(10):
#    var = X_test[:,i]
#    m_v = np.sum(var)/len(var)
#    s_v = np.sqrt(np.sum((var-(np.sum(var)/len(var)))**2)/len(var))
#    X_test[:,i] = (var-m_v)/s_v       #Data is overwritten in "X" variable

lda = LDA(n_components=2)
lda_transformed = pd.DataFrame(lda.fit_transform(X,y))

for i in range(11):
    plt.scatter(lda_transformed[y==i][0], lda_transformed[y==i][1])

plt.legend(loc=3)
plt.show()