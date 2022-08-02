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
oob_score = []
fo = open('Code_output.txt','w')
start = 7
stop = 20
step = 2
j = 0

for i in range(start, stop,step):
    forest = RandomForestClassifier(max_features = i)
    forest.fit(X_train, y_train)

    accuracy = forest.score(X_test, y_test)
    oob_acc = forest.oob_score()
    acc.append(accuracy)
    oob_score.append(oob_acc)
    print ('The val_score is', 100*accuracy, '% on the test data with number of features = ', i)
    print ('The oob_score is', 100*oob_acc, '% on the test data with number of features = ', i)

    fo.write ('\n'+'The val_score is '+str( 100*acc[j])+ '% on the test data with number of features = '+str(i))
    fo.write ('\n'+'The oob_score is '+str( 100*oob_score[j])+ '% on the test data with number of features = '+str(i))

    j= j+1
fo.close()

val_error = []
for h in range(len(acc)):
    val_error.append(100-100*acc[h])
    
fig, oob = plt.subplots()
fig.set_size_inches(14, 8)
val = oob.twinx()

oob.set_title("OOB Error vs Test Error")
oob.set_xlabel("Number of Features")
oob.set_ylabel("OOB Error")
val.set_ylabel("Test Error")

oob_lines = oob.plot(range(start,stop,step), [oob_score[k]*100 for k in range(len(oob_score))], 'r')
val_lines = val.plot(range(start,stop,step), val_error, 'c')
all_lines = oob_lines + val_lines
oob.legend(all_lines, ["OOB Error", "Test Error"])

fig.savefig("plot.pdf")