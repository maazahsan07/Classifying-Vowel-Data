# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

v_test = pd.read_csv('vowel_test.csv').iloc[:,1:13]
v_train = pd.read_csv('vowel_train.csv').iloc[:,1:13]

X = v_train.iloc[:,1:11]
y = v_train.iloc[:,0]

X_norm = (X - X.min())/(X.max() - X.min())

lda = LDA(n_components=2) #2-dimensional LDA
lda_transformed = pd.DataFrame(lda.fit_transform(X_norm, y))


# Plot all three series
for i in range(5):
    plt.scatter(lda_transformed[y==i][0], lda_transformed[y==i][1], label='Class 1')

# Prettify the graph
plt.legend()
plt.legend(loc='upper left');
plt.show()