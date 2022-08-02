        # -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 23:22:51 2021

@author: Admin
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from randomforest import RandomForestClassifier
from csv import reader
import random
from pandas import DataFrame as df
import matplotlib.pyplot as plt

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

with open("spam.data.txt",'r') as f:
            plaintext = f.read()
plaintext = plaintext.replace(' ',',')    
with open("spam.data.csv",'w') as f:
    f.write(plaintext)

dataset = list()
with open("spam.data.csv", 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        dataset.append(row)
    print ("Number of records: %d" % len(dataset))
    random.shuffle(dataset)
# convert string attributes to integers
for i in range(0, len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
df1 = df(dataset)
y1 = df1[57]
del df1[57]
X1  = df1
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1, test_size=0.3, random_state=1234)
X_train = X_train1.to_numpy()
X_test = X_test1.to_numpy()  
y_train = y_train1.to_numpy()
y_test = y_test1.to_numpy()
acc = []
fo = open('Code_output.txt','w')
start = 7
stop = 58
step = 5
j = 0

for i in range(start, stop,step):
    forest = RandomForestClassifier(max_features = i)
    forest.fit(X_train, y_train)

    accuracy = forest.score(X_test, y_test)
    acc.append(accuracy)
    print ('The accuracy was', 100*accuracy, '% on the test data with number of features = ', i)
    fo.write ('\n'+'The accuracy was '+str( 100*acc[j])+ '% on the test data with number of features = '+str(i))
    j= j+1
fo.close()

plt.plot(range(start,stop,step), acc)
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.title('Specificity wrt number of features')
