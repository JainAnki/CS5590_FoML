# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 11:20:54 2021

@author: Ankita Jain
"""

from numpy import genfromtxt
from sklearn import svm, metrics
from csv import reader
import numpy as np
import pandas as pd

def parse_G(file_X, file_y):
    for file in [file_X, file_y]:
        with open(file) as in_file:
        
            # create a csv reader object
            csv_reader = reader(in_file)
            data = []
            # go over each line 
            for line in csv_reader:
        
                # if line is not empty
                if line:
                    data_ = line[0].split(' ')
                    data.append(data_)
            if file == file_X:
                df = pd.DataFrame(data).apply(pd.to_numeric)
            else:
                y = pd.DataFrame(data).apply(pd.to_numeric)
    return df[df.columns[0:5000]],np.ravel(y)

def parse(file):
    headers = ['Digit' ,'Feat_1', 'Feat_2']
    X = np.matrix(genfromtxt(file))
    df = pd.DataFrame(X, columns = headers).apply(pd.to_numeric)
    df =  df[df['Digit'].isin([1,5])]
    y_ = df['Digit']
    X_ = df[['Feat_1', 'Feat_2']]    
    return X_,y_

def accuracy(X_train, y_train, X_test, y_test, c, kernel, Q, g, coef):
    if kernel == 'linear':
        clf = svm.SVC(kernel = 'linear')
    elif kernel == 'poly':
        clf = svm.SVC(C = c, kernel = 'poly', degree = Q, coef0 = coef)
    elif kernel == 'rbf':
        clf = svm.SVC(C = c, kernel = 'rbf', gamma = g)

    clf.fit(X_train,y_train)
    y_tr_pred = clf.predict(X_train)
    y_te_pred = clf.predict(X_test)
    tr_error = (1- metrics.accuracy_score(y_train, y_tr_pred))
    te_error = (1- metrics.accuracy_score(y_test, y_te_pred))
    no_= clf.support_.size
    return tr_error, te_error, clf,  no_

X_, y_ = parse('features_train.txt')
X_test, y_test = parse('features_test.txt')

_,te_error_,_,no_ = accuracy(X_,y_,X_test,y_test, 0, 'linear', 0,0,0)
print("Accuracy: ", (1-te_error_))
fo = open('Code_output.txt','w')
fo.write("4a" +"\n"+"Kernel: linear"+"\n"+"Number of Support Vectors: "+ str(no_)+"\n"+"Test Accuracy: "+ str(1-te_error_)+"\n")

fo.write("\n"+"4b"+"\n"+"Kernel: linear")
no_ = [50, 100, 200, 800]
print("\n" + "4b")
for x in no_:
    X= X_[0:x]
    y = y_[0:x]

    _,te_error_,_,no_ = accuracy(X,y,X_test,y_test, 0, 'linear', 0,0,0)
    print("Accuracy for %0.0f training samples:" %x, (1-te_error_))
    print("The number of Support Vectors are ", no_)
    fo.write("\n"+"\n"+"No. of training samples: "+ str(x)+"\n"+"Number of Support Vectors: "+ str(no_)+"\n"+"Test Accuracy: "+ str(1-te_error_))
#####################################################
#                                                   #
#               POLYNOMIAL KERNEL                   #
#                                                   #
#####################################################

print("\n"+"\n" + "4c")

tr_error_2,_,_,_ = accuracy(X_,y_,X_test,y_test, 0.0001, 'poly', 2,1,1)
tr_error_5,_,_,_ = accuracy(X_,y_,X_test,y_test, 0.0001, 'poly', 5,1,1)
i = 'TRUE' if tr_error_5 > tr_error_2 else 'FALSE'
print("i: ", i)
_,_,_,no_2 = accuracy(X_,y_,X_test,y_test, 0.001, 'poly', 2,1,1)
_,_,_,no_5 = accuracy(X_,y_,X_test,y_test, 0.001, 'poly', 5,1,1)
ii = 'TRUE' if no_5 < no_2 else 'FALSE'
print("ii: ", ii)
tr_error_2,_,_,_ = accuracy(X_,y_,X_test,y_test, 0.01, 'poly', 2,1,1)
tr_error_5,_,_,_ = accuracy(X_,y_,X_test,y_test, 0.01, 'poly', 5,1,1)
iii = 'TRUE' if tr_error_5 > tr_error_2 else 'FALSE'
print("iii: ", iii)
_,te_error_2,_,_ = accuracy(X_,y_,X_test,y_test, 1, 'poly', 2,1,1)
_,te_error_5,_,_ = accuracy(X_,y_,X_test,y_test, 1, 'poly', 5,1,1)
iv = 'TRUE' if te_error_5 < te_error_2 else 'FALSE'
print("iv: ", iv)
fo.write("\n"+"4c" +"\n"+ "Kernel: poly"+"\n"+"i: "+ str(i)+"\n"+"ii: "+ str(ii)+"\n"+"iii: "+ str(iii)+"\n"+"iv: " +str(iv)+"\n")

#####################################################
#                                                   #
#                    RBF KERNEL                     #
#                                                   #
#####################################################

c_ = [0.01, 1, 100, 10**4, 10**6]
tr = []
te = []
print("\n" + "4d")

fo.write("\n"+"4d" +"\n")
fo.write("Kernel: rbf"+"\n")

for x in c_:
    tr_error,te_error,_,_ = accuracy(X_,y_,X_test,y_test, x, 'rbf', 0,1,0)    

    tr.append(tr_error)
    te.append(te_error)
    print("C = %0.02f    Training error = %0.08f    Test_error = %0.08f" %(x, tr_error, te_error)) 

    fo.write("C: "+ str(x)+"\n"+"Train Error: "+ str(tr_error)+"\n"+"Test Error: "+ str(te_error)+"\n"+"\n")

min_tr = [i for  i,x in enumerate(tr) if x == min(tr)]
min_te = [i for i,x in enumerate(te) if x == min(te)]

c_tr = [x_tr for j,x_tr in enumerate(c_) if min_tr[0] == j]
c_te = [x_te for j,x_te in enumerate(c_) if min_te[0] == j]

print("Min train error corresponds to C = " ,c_tr)
print("Min test error corresponds to C = " , c_te)
fo.write("Min train error corresponds to C =  "+ str(c_tr)+"\n"+"Min test error corresponds to C =  "+ str(c_te)+"\n")

X_G, y_g = parse_G('gisette_train.data', 'gisette_train.labels')
X_Gtest, y_gtest = parse_G('gisette_valid.data', 'gisette_valid.labels')

tr_error_l, te_error_l, clf_l, no_l = accuracy(X_G,y_g,X_Gtest,y_gtest, 0 ,'linear', 0,0,0)    

print("\n" + "5a")
print("Train error = " ,(tr_error_l))
print("Test error = " , (te_error_l))
print("Accuracy for %0.04f training samples"%len(X_G), (1-te_error_l))
print("The number of Support Vectors are ",no_l)
fo.write("\n"+"5a" +"\n"+"Kernel: linear"+"\n"+"Number of Support Vectors: "+ str(no_l)+"\n"+"Train Error: "+ str(tr_error_l)+"\n"+"Test Error: "+ str(te_error_l)+"\n")

tr_error_r, te_error_r, clf_r, no_r = accuracy(X_G,y_g,X_Gtest,y_gtest, 1.0 ,'rbf', 0,0.001,0)    

print("\n" + "5b")
print("Train error RBF= " ,(tr_error_r))
print("Test error RBF = " ,(te_error_r))
print("Accuracy for RBF %0.04f training samples"%len(X_Gtest), (1-te_error_r))
print("The number of Support Vectors for RBF are ",no_r)

fo.write("\n"+"5b" +"\n"+"Kernel: rbf"+"\n"+"Number of Support Vectors: "+ str(no_r)+"\n"+"Train Error: "+ str(tr_error_r)+"\n"+"Test Error: "+ str(te_error_r)+"\n")

tr_error_p, te_error_p, clf_p, no_p = accuracy(X_G,y_g,X_Gtest,y_gtest, 1.0 ,'poly', 2,1,1)    

print("Train error Polynomial = " ,tr_error_p)
print("Test error Polynomial = " , te_error_p)
print("Accuracy for Polynomial %0.04f training samples"%len(X_Gtest), (1-te_error_p))
print("The number of Support Vectors for Polynomial are ",no_p)
fo.write("\n"+"Kernel: poly"+"\n"+"Number of Support Vectors: "+ str(no_p)+"\n"+"Train Error: "+ str(tr_error_p)+"\n"+"Test Error: "+ str(te_error_p)+"\n")

tr = [tr_error_r, tr_error_p]
te = [te_error_r, te_error_p]

min_tr_G = [[i,x] for  i,x in enumerate(tr) if x == min(tr)]
min_te_G = [[i,x] for  i,x in enumerate(te) if x == min(te)]
for a in [min_tr_G, min_te_G]:
    if a[0][0] == 0:
        a[0][0] = 'RBF'   
    if a[0][0] == 1:
        a[0][0] = 'Poly'

print("MIN TRAINING ERROR: ", min_tr_G )
print("MIN TESTING ERROR: ", min_te_G)
fo.write("\n"+"MIN TRAINING ERROR: "+ str(min_tr_G)+"\n"+"MIN TESTING ERROR: "+ str(min_te_G)+"\n")
fo.close()