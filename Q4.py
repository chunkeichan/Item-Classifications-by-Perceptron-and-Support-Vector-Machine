"""
Created on Tue Apr 17 2018

@author: JQ

Modified on Mon Feb 22 2021

HC Wu

"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from sklearn import svm

Y_true = [0,1,1,0,0,0,1,1,0,1]

fcf = sorted(glob.glob('data/*.txt')) # list .txt files only
file = open(fcf[0], 'r')
text1 = file.read() # string
file.close()

file = open(fcf[50], 'r')
features = file.read().split()

# Training data
Xtrain = np.zeros((1,np.shape(features)[0])) # empty list
for text_index in range(40):
    file_temp = open(fcf[text_index], 'r')
    text_temp = file_temp.read()
    occurence_temp = []
    for feature_index in features:
        occurence_temp.append(text_temp.count(feature_index)) 
    Xtrain = np.concatenate((Xtrain, [occurence_temp]), axis = 0) # shape: 41-by-13

Xtrain = np.delete(Xtrain, (0), axis = 0) # remove the first empty row 

# Testing data
Xtest = np.zeros((1,np.shape(features)[0])) # empty list
for text_index in range(40,50):
    file_temp = open(fcf[text_index], 'r')
    text_temp = file_temp.read()
    occurence_temp = []
    for feature_index in features:
        occurence_temp.append(text_temp.count(feature_index)) 
    Xtest = np.concatenate((Xtest, [occurence_temp]), axis = 0) # shape: 11-by-13
    
Xtest = np.delete(Xtest, (0), axis = 0) # remove the first empty row

#XTX = np.dot(Xtrain.T, Xtrain)


# Subtracting from the mean and scaling are not required. But it is acceptable if you did so.
meanX = np.mean(Xtrain,axis=0)
CtrX = Xtrain-meanX
CtrXtest = Xtest-meanX
S1 = np.dot(np.transpose(CtrX),CtrX)
XTX = 1.0/(Xtrain.shape[0]-1)*S1

w, v = LA.eig(XTX) # compute eigenvalues and eigenvectors

# Due to the descending order of eigenvalues in w, here we choose the first two columns of v.
PC = v[:,0:2] # principal comonents; shape: 13-by-2
t = np.dot(CtrX,PC) # PC scores for training data; shape: 40-by-2
tt = np.dot(CtrXtest,PC) # PC scores for testing data; shape: 10-by-2 

# ==============  Perceptron =============================
"""
        You may refer to pg. 26, Ch2 for the perceptron example.    
"""
X1 = t[0:20][:]
X2 = t[20:40][:]

Y1 = np.concatenate((np.ones((20,1)),X1), axis=1)
Y2 = np.concatenate((np.ones((20,1))*-1,-X2), axis=1)
Yt = np.concatenate((Y1,Y2), axis=0)  # shape: 40-by-3

# Initialize
a = np.zeros((3,1))

# no. of misclassified samples
sum_wrong = 1

a_iter = a
k = 0

while sum_wrong > 0 and k < 1000:
    wrong = np.dot(Yt,a_iter) <= 0
    sum_wrong = sum(wrong)
    sum1 = sum(wrong*np.ones((1,3))*Yt)
    a_iter = a_iter+sum1.reshape(3,1)
    k=k+1
    
a_con = a_iter

Y_perceptron = []
for i in range(10):
    w_temp = np.array([np.append(1, tt[i][:])]).T # (3,1)
    J = np.dot(a_con.T, w_temp)
    if J > 0:
        Y_perceptron.append(0)
    else:
        Y_perceptron.append(1)

error_perceptron_projected = sum(sum(abs(np.array([Y_true]) - np.array([Y_perceptron]))))/10
print("Error rate for projected test samples by perceptron", error_perceptron_projected)
        


# ============== Hard-margin SVM =========================
"""
        You may refer to pg. 20, Ch3 for the hard-margin SVM example.    
"""
X1 = t[0:20][:]
X2 = t[20:40][:]
X = np.concatenate((X1,X2), axis=0)
Y_svm=np.concatenate((np.zeros((1,20)),np.ones((1,20))), axis=1)
Y_svm=Y_svm.ravel()

#Fit Linear SVM
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, Y_svm)

"""
       Q4 Section 1: Put down your answers here
"""
Y_SVM_pred=clf.predict(tt)

"""
        End of Q4 Section 1    
"""
error_SVM_PCA = sum(sum(abs(np.array([Y_true]) - np.array([Y_SVM_pred]))))/10
print("Error rate for projected test samples by SVM", error_SVM_PCA)



# ==============  Perceptron =============================
"""
        You may refer to  Ch2 for the perceptron example.    
"""
X1 = Xtrain[0:20][:]
X2 = Xtrain[20:40][:]

Y1 = np.concatenate((np.ones((20,1)),X1), axis=1)
Y2 = np.concatenate((np.ones((20,1))*-1,-X2), axis=1)
Yt = np.concatenate((Y1,Y2), axis=0)  # shape: 40-by-3

# Initialize
a = np.zeros((14,1))

# no. of misclassified samples
sum_wrong = 1

a_iter = a
k = 0

while sum_wrong > 0 and k < 1000:
    wrong = np.dot(Yt,a_iter) <= 0
    sum_wrong = sum(wrong)
    sum1 = sum(wrong*np.ones((1,14))*Yt)
    a_iter = a_iter+sum1.reshape(14,1)
    k=k+1
    
a_con = a_iter

Y_perceptron = []
for i in range(10):
    w_temp = np.array([np.append(1, Xtest[i][:])]).T # (3,1)
    J = np.dot(a_con.T, w_temp)
    if J > 0:
        Y_perceptron.append(0)
    else:
        Y_perceptron.append(1)

error_perceptron = sum(sum(abs(np.array([Y_true]) - np.array([Y_perceptron]))))/10
print("Error rate for  test samples by perceptron", error_perceptron)
        


# ============== Hard-margin SVM =========================
"""
        You may refer to pg. 20, Ch3 for the hard-margin SVM example.    
"""
X1 = Xtrain[0:20][:]
X2 = Xtrain[20:40][:]
X = np.concatenate((X1,X2), axis=0)

Y_svm_all=np.concatenate((np.zeros((1,20)),np.ones((1,20))), axis=1)
Y_svm_all=Y_svm_all.ravel()

#Fit Linear SVM
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, Y_svm_all)

"""
       Q4 Section 2: Put down your answers here
"""           
Y_SVM_pred=clf.predict(Xtest)

"""
       End of Q4 Section 2
"""
error_SVM = sum(sum(abs(np.array([Y_true]) - np.array([Y_SVM_pred]))))/10

print("Error rate for test samples by SVM", error_SVM)




