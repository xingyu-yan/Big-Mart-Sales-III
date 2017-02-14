'''
@author: xingyu, created on December 5, 2016, at Ecole Centrale de Lille
https://github.com/xingyu-yan
# This programme is for Big Mart Sales Practice Problem with ANN
# Web source URL: https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/
# ANN method reference: Coursera Machine Learning open course (Andrew Ng)
'''

import numpy as np
import pandas as pd

# Loading data
print("Loading Data ...\n")
DataTrain = pd.read_csv('Train_UWu5bXk.csv') 
A_col_train = np.matrix(DataTrain)
#print(A_col_train)
m = len(A_col_train)
#print(m)

Item_Identifier = A_col_train[0:m,0]
Item_Weight = A_col_train[0:m,1]
#print(Item_Weight[0:8])
Item_Fat_Content = A_col_train[0:m,2]
Item_Visibility = A_col_train[0:m,3]
Item_Type = A_col_train[0:m,4]              
Item_MRP = A_col_train[0:m,5]               
Outlet_Identifier = A_col_train[0:m,6]      
Outlet_Establishment_Year = A_col_train[0:m,7]
Outlet_Size = A_col_train[0:m,8] 
#print(Outlet_Size[0:5])
Outlet_Location_Type = A_col_train[0:m,9]  
Outlet_Type = A_col_train[0:m,10]

y_train = A_col_train[0:m,11]

t = np.math.isnan(Outlet_Size[3])
#print(t)

# Data processing 
Item_weight = np.zeros((m,1))
for i in range(m):
    tt = Item_Weight[i]
    if np.math.isnan(tt):
        ItemIdentifier = Item_Identifier[i]
        for j in range(m):
            identifier = Item_Identifier[j]
            if identifier == ItemIdentifier:
                weight = Item_Weight[j]
                if np.math.isnan(weight) == False:
                    ttt = Item_Weight[j]
                    break
        t = ttt 
    else:
        t = tt
    Item_weight[i] = t        
print(Item_weight[0:8])

Outlet_type = np.zeros((m,1))
for numb in range(m):
    t = Outlet_Type[numb]
    if t == 'Grocery Store':
        Outlet_type[numb] = 1
    elif t == 'Supermarket Type1':
        Outlet_type[numb] = 2
    elif t == 'Supermarket Type2':
        Outlet_type[numb] = 3
    elif t == 'Supermarket Type3':
        Outlet_type[numb] = 4
    else:  
        print('This is an extra lable.')
print(Outlet_type)    

Outlet_location_type = np.zeros((m,1))
for numb in range(m):
    t = Outlet_Location_Type[numb]
    if t == 'Tier 1':
        Outlet_location_type[numb] = 1
    elif t == 'Tier 2':
        Outlet_location_type[numb] = 2
    elif t == 'Tier 3':
        Outlet_location_type[numb] = 3
    else:  
        print('This is an extra lable.')
print(Outlet_location_type)       
    
Item_fat_content = np.zeros((m,1))
for numb in range(m):
    t = Item_Fat_Content[numb]
    if t == 'LF':
        Item_fat_content[numb] = 0
    elif t == 'Low Fat':
        Item_fat_content[numb] = 0
    elif t == 'low fat':
        Item_fat_content[numb] = 0
    elif t == 'Regular':
        Item_fat_content[numb] = 1
    elif t == 'reg':
        Item_fat_content[numb] = 1
    else:  
        print('This is an extra lable.')
print(Item_fat_content)

Item_identifier = np.zeros((m,1))
C = np.unique([Item_Identifier]) 
#print(C)   
lenC = len(C)
for numb in range(m):
    t = Item_Identifier[numb]
    for j in range(lenC):
        tt = C[j]
        if t == tt:
            Item_identifier[numb] = j
            break
print(Item_identifier)

Item_type = np.zeros((m,1))
C = np.unique([Item_Type]) 
lenC = len(C)
for numb in range(m):
    t = Item_Type[numb]
    for j in range(lenC):
        tt = C[j]
        if t == tt:
            Item_type[numb] = j
            break
print(Item_type)

Outlet_identifier = np.zeros((m,1))
C = np.unique([Outlet_Identifier]) 
lenC = len(C)
for numb in range(m):
    t = Outlet_Identifier[numb]
    for j in range(lenC):
        tt = C[j]
        if t == tt:
            Outlet_identifier[numb] = j
            break
print(Outlet_identifier)

Item_visibility = np.zeros((m,1))
Item_mrp = np.zeros((m,1))
Outlet_establishment_year = np.zeros((m,1))
for i in range(m):
    Item_visibility[i] = Item_Visibility[i]
    Item_mrp[i] = Item_MRP[i]
    Outlet_establishment_year[i] = Outlet_Establishment_Year[i]
        
print('All the Orginal Trainning Data:')
Data_Train_Orginal = np.hstack((Item_identifier, Item_weight, Item_fat_content, Item_visibility, 
                      Item_type, Item_mrp, Outlet_identifier, Outlet_establishment_year, 
                      Outlet_location_type, Outlet_type, y_train))
print(Data_Train_Orginal)

# Save data after data processing 
import csv
b = open('TrainDataOrginal.csv', 'a') 
a = csv.writer(b)
a.writerows(Data_Train_Orginal)  
b.close

# Data normalization and normalized data saving
DTOsize = np.shape(Data_Train_Orginal)
#print(DTOsize)
RowSize = DTOsize[0]
ColumnSize = DTOsize[1]
#print(DTOsize[1])
#print(DTOsize[0])

Data_Train_Normalized = np.zeros((RowSize,ColumnSize))
for j in range(ColumnSize):
    maxValue = max(Data_Train_Orginal[:,j])
    minValue = min(Data_Train_Orginal[:,j])
    for i in range(RowSize):
        Data_Train_Normalized[i,j] = (Data_Train_Orginal[i,j]-minValue)/(maxValue-minValue)
#print('All the Normalized Trainning Data:')
#print(Data_Train_Normalized)
b = open('TrainDataNormalized.csv', 'a') 
a = csv.writer(b)
a.writerows(Data_Train_Normalized)  
b.close
print('Finished')
