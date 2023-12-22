# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 00:41:03 2019

@author: princ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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