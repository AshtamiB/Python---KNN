# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 03:19:59 2018

@author: Ashtami
"""
#EXAMPLE 1

from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
#----------------- Generate Synthetic Data ---------------#
X_R, y_R = make_regression(n_samples=100, n_features=1, n_informative=1,
bias=150.0, noise=30)
fig, subaxes = plt.subplots(5, 1 , figsize=(11,8), dpi=100)
X = np.linspace(-3, 3, 500).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X_R,
y_R,random_state=0)
#--------------------------- KNN -------------------------#
for K, K in zip(subaxes,[1, 3, 7, 15, 59]):
 knn_reg = neighbors.KNeighborsRegressor(n_neighbors=K)
 knn_reg.fit(X_train, y_train)
 y_predict_output = knn_reg.predict(X)
 plt.plot(X, y_predict_output)
 plt.plot(X_train, y_train, 'o', alpha=0.9, label='Train')
 plt.plot(X_test, y_test, '^', alpha=0.9, label='Test')
 plt.xlabel('Input feature')
 plt.ylabel('Target value')
 plt.title('KNN Regression (K={})\n$'.format(K))
 plt.legend()
 plt.show()
 
 #EXAMPLE 2
 
 import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skp
from sklearn import neighbors
X = np.arange(-10,10).reshape(-1, 1)
y = np.sinc(0.5*X)
knn_reg = neighbors.KNeighborsRegressor(2, weights='uniform')
knn_reg.fit(X, y)
s = knn_reg.predict(X)
plt.plot(X, y, label='real')
plt.plot(X, s, label='KNN Reg')
plt.legend()
plt.show()