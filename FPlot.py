#!/usr/local/bin/python3.5
# encoding: utf-8
'''
Created on January 28, 2016
@author: Ran CHEN & Xingyu YAN, at University of Lille1
# This programme is for Big Mart Sales Practice Problem with ANN
# Web source URL: https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/
# ANN method reference: Coursera Machine Learning open course (Andrew Ng)
'''

import numpy as np
import pandas as pd
'''import scipy.io as sio
from BMS_package.BMSfuncs import nnCostFunction, sigmoidGradient, randInitializeWeights, checkNNGradients, cgbt, predict
from BMS_package.BMSfuncs import print_results'''
import matplotlib.pyplot as plt

# Part 1: Loading and visualizing data

print("Loading Data ...\n")
myData = pd.read_csv('TrainDataNormalized.csv') 
TrainData = np.matrix(myData)
print(TrainData.shape)
X0 = TrainData[:,0]
X1 = TrainData[:,1]
X2 = TrainData[:,2]
X3 = TrainData[:,3]
X4 = TrainData[:,4]
X5 = TrainData[:,5]
X6 = TrainData[:,6]
X7 = TrainData[:,7]
X8 = TrainData[:,8]
X9 = TrainData[:,9]
X10 = TrainData[:,10]

fig = plt.figure(1)
plt.subplot(341)
plt.scatter(X0, X10)
plt.subplot(342)
plt.scatter(X1, X10)
plt.subplot(343)
plt.scatter(X2, X10)
plt.subplot(344)
plt.scatter(X3, X10)
plt.subplot(345)
plt.scatter(X4, X10)
plt.subplot(346)
plt.scatter(X5, X10)
plt.subplot(347)
plt.scatter(X6, X10)
plt.subplot(348)
plt.scatter(X7, X10)
plt.subplot(349)
plt.scatter(X9, X10)
plt.subplot(3,4,10)
plt.scatter(X9, X10)
plt.show()