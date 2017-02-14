'''
Created on January 28, 2016
@author: Ran CHEN & Xingyu YAN, at University of Lille1 & Ecole Centrale de Lille
https://github.com/xingyu-yan
# This programme is for Big Mart Sales Practice Problem with ANN
# Web source URL: https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/
# ANN method reference: Coursera Machine Learning open course (Andrew Ng)
'''

import numpy as np
import pandas as pd
from BMS_package.BMSfuncs import randInitializeWeights, trainNN, predict
from BMS_package.BMSfuncs import print_results
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

X_train = np.hstack((X3,X5,X9))
print(X_train.shape)
y_train = X10
print(y_train.shape)

input_layer_size = 3   
hidden_layer_size = 10  
num_labels = 1

# Part 6: Initializing parameters
initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels)

initial_nn_params = np.r_[np.reshape(initial_Theta1,hidden_layer_size*(input_layer_size+1),order='F'),np.reshape(initial_Theta2,num_labels*(hidden_layer_size+1),order='F')]

# Part 9: Training NN
lamb = 0.04

Theta = trainNN(initial_nn_params,X_train,y_train,input_layer_size,hidden_layer_size,num_labels,lamb,0.25,0.5,5,1e-8)

Theta1 = np.matrix(np.reshape(Theta[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1),order='F'))
Theta2 = np.matrix(np.reshape(Theta[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1),order='F'))

# Part 11: Implement predict
p = predict(Theta1,Theta2,X_train).T
print_results('train_set', p-y_train)

#Plot the results
'''fig = plt.figure(1)
plt.plot(p[7900:8000], 'b--', label = 'predict')
plt.plot(y_train[7900:8000], 'g', label = 'real')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('some examples of forecasting results')
ax.set_xlabel('Number')
ax.set_ylabel('Sales (per unit)')
plt.legend()
plt.show()'''

fig = plt.figure(1)
plt.plot(p[7900:8000], 'b--', label = 'predict')
plt.plot(y_train[7900:8000], 'g', label = 'real')
plt.title('some examples of forecasting results')
plt.xlabel('Number')
plt.ylabel('Sales (per unit)')
plt.legend()
plt.show()

print("Loading Test Data ...\n")
myData = pd.read_csv('TestDataNormalized.csv') 

TestData = np.matrix(myData)
print(TrainData.shape)
X0 = TestData[:,0]
X1 = TestData[:,1]
X2 = TestData[:,2]
X3 = TestData[:,3]
X4 = TestData[:,4]
X5 = TestData[:,5]
X6 = TestData[:,6]
X7 = TestData[:,7]
X8 = TestData[:,8]
X9 = TestData[:,9]

X_test = np.hstack((X3,X5,X9))
print(X_train.shape)

#predict test data 
p = predict(Theta1,Theta2,X_test).T
print(p.shape)
#Reverse the normalized test set data
DataTrain = pd.read_csv('Train_UWu5bXk.csv') 
A_col_train = np.matrix(DataTrain)
m = len(A_col_train)
y_train = A_col_train[0:m,11]
min_y_train = min(y_train)
max_y_train = max(y_train)
predTest_reverseNormalized = np.zeros(len(p))
for i in range(len(p)):
    predTest_reverseNormalized[i] = p[i]*(max_y_train-min_y_train)+min_y_train
print('Program is finished here')    
'''import csv
b = open('predTest.csv', 'a') 
a = csv.writer(b)
a.writerows(predTest_reverseNormalized)  
b.close'''
