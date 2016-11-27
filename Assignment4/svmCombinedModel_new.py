# -*- coding: utf-8 -*-
#######################################################################
# Function to implement one-vs-all SVM
# For SVM we use the scikit learn's SVM implementation
#######################################################################
from __future__ import division
import pdb
import itertools

import numpy as np
from scipy import stats
from sklearn.svm import SVC
import matplotlib.pyplot as plt

plt.figure()
plt.ion()

def cost(W,X,Y,C):
    """
    W = [K,D] K = number of classes, D = number of dimensions
    Y = true classes {1,...,10}
    """
    [K,D] = W.shape 
    [N,D] = X.shape
    if not hasattr(cost, "negInfMask"):
        cost.negInfMask = np.arange(N)*K + (Y)
        # Y values are in {1,...,10} but array indices
        # in python are in {0,...,9}

    normW = 0.5 * np.sum(np.linalg.norm(W,axis=1)**2)
    # wYiXi = np.diag(np.dot(W[Y-1,:], X.T))
    # wYiXi = np.einsum('ij,ji->i',W[Y-1,:], X.T)
    wYiXi = np.einsum('ij,ij->i',W[Y.astype(np.int32),:], X)
    wJXi  = np.dot(X,W.T) 
    # wJXi  = np.einsum('ij,kj->ik',X,W) # here einsum is slower
    # We mask j = Yi for each i with -np.inf
    # before that we need to convert (i,Yj) to 
    # row major array index
    np.put(wJXi, cost.negInfMask, -np.inf)
    max_wJXi = np.max(wJXi, axis=1)

    slack = 1 - wYiXi + max_wJXi
    costVal = normW + C * np.sum(np.maximum(0, slack))  
    # return (costVal, slack) 
    return costVal 

def subgradient(W,xi,yi,sum_X,C,N):
    """
    sum_X = [1,D] (not used anymore)
    """
    [K,D] = W.shape
    
    Wxi = np.dot(W,xi.T)
    Wxi_yi = np.copy(Wxi[yi])
    Wxi[yi]=-np.inf # in next step we want argmax j!=yi
    j_star = np.argmax(Wxi)
    # print("j_star {}".format(j_star))

    Ninv = 1.0 / N

    deltaW = np.copy(W) * Ninv

    if 1 - Wxi_yi + Wxi[j_star] >= 0:
        for j in xrange(K):
            if j == yi:
                # deltaW[j,:] = W[j,:] * Nin - C*xi
                deltaW[j,:] -= C*xi
            elif j == j_star:
                deltaW[j,:] += C*xi
            else:
                # deltaW[j,:] == W[j,:] from initialization
                # of deltaW
                pass
        # pdb.set_trace()
    #deltaW /= np.linalg.norm(deltaW,2,axis=1).reshape(K,1)
    
    # gradient clipping
    # np.clip(deltaW,-C,C,out=deltaW);

    return deltaW

def svmCombinedModel(X_train, X_test, y_train, C = 10, iters = 15000):

    [N,D]  = X_train.shape
    [M,Dt] = X_test.shape 

    assert D == Dt, "Train and Test data must have the same number of features (dimensions)"
    # If we don't have a particular class in the training set
    # Then we can't learn to classify it. Using this logic 
    # we get the set of K - classes which we need to train as
    # follows.
    labels = np.unique(y_train)
    K = len(labels)
    
    # Random initilization of the Weight matrix [K,D]
    W = np.random.random((K,D))
    costList = np.zeros((iters+1, 1))
    costList[0] = cost(W,X_train,y_train,C)
    
    sum_X = np.sum(X_train, axis=0)
    for iterCnt in xrange(iters):
        iterCnt += 1
        randIdx = np.random.randint(N)
        xi,yi = X_train[randIdx, :], y_train[randIdx]
        deltaW = subgradient(W, xi, yi, sum_X, C, N)
        tk = 1.0/(10 * C * (iterCnt ** 0.25))
        # tk = 1.0
        W -= tk * deltaW
        # W /= np.linalg.norm(W,2,axis=1).reshape(K,1)
        costList[iterCnt] = cost(W,X_train,y_train,C)
        if iterCnt % 1000 == 0:
            print("{} - Cost is : {}".format(iterCnt, np.mean(costList[iterCnt-10:iterCnt])))
            ytrain_pred = np.dot(X_train, W.T)
            ytrain_pred = np.argmax(ytrain_pred, axis=1)
            print(np.sum(ytrain_pred == y_train))


    ytest_pred  = np.dot(X_test, W.T) 
    ytrain_pred = np.dot(X_train, W.T)

    ytest_pred = np.argmax(ytest_pred, axis=1)  
    ytrain_pred = np.argmax(ytrain_pred, axis=1)

    assert ytrain_pred.shape == (N,), "Count of training predictions must match the number of examples in training dataset "
    assert ytest_pred.shape == (M,), "Count of test predictions must match the number of examples in test dataset "

    return (ytest_pred, ytrain_pred, costList)


if __name__ == "__main__":

    np.random.seed(12345)

    # Load data
    X_train = np.loadtxt("./data/X_train.txt")
    X_test  = np.loadtxt("./data/X_test.txt")
    y_train = np.loadtxt("./data/y_train.txt").astype(np.int32) - 1
    y_test  = np.loadtxt("./data/y_test.txt").astype(np.int32) - 1

    iters = 15000

    [ytest_pred, ytrain_pred, costList] = svmCombinedModel(X_train, X_test, y_train, C=10, iters=iters)
    print(np.sum(ytest_pred == y_test))
    print(np.sum(ytrain_pred == y_train))

    plt.scatter(range(len(costList)-1), costList[1:])
    plt.xlim([0,iters])
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()
    pdb.set_trace()
