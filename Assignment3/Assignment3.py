from __future__ import division
# coding: utf-8

# Evaluate performance of algorithm on a fixed test dataset based 
# on stochastic dual gradient ascent $\texttt{svmTrainDCGA.py}$ 
# and $\texttt{svmTrainQP.y}$ for different training size 
# $N_{tr} = [100,500,1000,2000,3000]$ for $C=1$

import sys
import time
import pdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from svmTrainDCGA2 import svmTrainDCGA 
from svmTrainQP import svmTrainQP

def svmTest(K_test_train, y, alpha):
    return np.sign(np.dot(K_test_train.T,y*alpha))

# load data
print("Loading data ..")
X_test  = np.loadtxt("X_test.txt")
# Adding in a column of ones, since we don't include a bias term
X_test  = np.concatenate((np.zeros((X_test.shape[0],1)),X_test),axis=1)

X_train = np.loadtxt("X_train.txt")
# Adding in a column of ones, since we don't include a bias term
X_train = np.concatenate((np.zeros((X_train.shape[0],1)),X_train),axis=1)

y_test  = np.loadtxt("y_test.txt")
y_train = np.loadtxt("y_train.txt")

# constants
C = 1
Ntr = [100,500,1000,2000,3000]

train_Nx, train_Nd = X_train.shape

def linear_kernel(X, Z):
    """
    X is (n,d) 
    Z is (m,d)
    
    Dimension of kernel_mat is (n,m)
    """
    n,d = X.shape
    m,d = Z.shape
    kernel_mat = np.zeros((n,m))
    for row in xrange(n):
        kernel_mat[row,:] = np.dot(Z,X[row,:]).T
    return kernel_mat

dcga_errors = []
qp_errors = []

dcga_time = []
qp_time = []

for ntr in Ntr:
    print("Number of datapoints : {}".format(ntr))
    # get ntr random indexes
    ntr_idxs = np.random.permutation(train_Nx)[:ntr]
    # get subset of data
    X_train_ = X_train[ntr_idxs,:]
    y_train_ = y_train[ntr_idxs]
    print("Computing Kernel ..")
    K_train_train = linear_kernel(X_train_, X_train_)
    K_test_train = linear_kernel(X_train_, X_test)

    print("Running SVM Train ..")
    timeSt = time.clock()
    a_dcga = svmTrainDCGA(K_train_train, y_train_, C)
    time1 = time.clock()
    a_qp = svmTrainQP(K_train_train, y_train_, C)
    time2 = time.clock()
    
    dcga_time.append(time1-timeSt)
    qp_time.append(time2-time1)

    print("DCGA time : {} QP time : {}".format(dcga_time[-1], qp_time[-1]))
    print("Testing ...")
    y_train_ = np.expand_dims(y_train_, axis=1)
    y_test_ = np.expand_dims(y_test, axis=1)
    # DCGA
    y_pred = svmTest(K_test_train, y_train_, a_dcga)
    dcga_errors.append(100 - (np.sum(y_pred != y_test_)*100/len(y_test_)))
    # QP
    y_pred = svmTest(K_test_train, y_train_, a_qp)
    qp_errors.append(100 - (np.sum(y_pred != y_test_)*100/len(y_test_)))

# Plotting code
plt.plot(Ntr,dcga_errors,'r-')
plt.plot(Ntr, qp_errors,'g-')
plt.xlabel("Number of datapoints.")
plt.ylabel("Accuracy in percent.")
plt.legend(["DCGA","QP"])


plt.figure()
plt.plot(Ntr, dcga_time,'r-')
plt.plot(Ntr, qp_time,'g-')
plt.xlabel("Number of datapoints.")
plt.ylabel("execution time (in seconds).")
plt.legend(["DCGA","QP"])

plt.show()
