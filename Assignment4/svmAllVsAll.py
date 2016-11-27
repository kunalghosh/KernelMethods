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

def svmAllVsAll(X_train, X_test, y_train, kernel, C):
    """
    Implements the one-vs-all classification using an SVM.

    params X_train: Input matrix with training data (Nxd)
    params X_test : Input matrix with the test data (Mxd)
    params y_train: Labels of the training examples (Nx1)
    params kernel : The kernel type. (string) one of the following:
                    [‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’]
                    source: sklearn documentation (http://tinyurl.com/lrpxw9k)
    params C      : The regularization paramters (scalar)
    
    returns ytest_pred : class prediction of test examples (Mx1)
    returns ytrain_pred: class prediction of training examples (Nx1)
    """
    
    [N,d]  = X_train.shape
    [M,dt] = X_test.shape 

    assert d == dt, "Train and Test data must have the same number of features (dimensions)"
    # If we don't have a particular class in the training set
    # Then we can't learn to classify it. Using this logic 
    # we get the set of K - classes which we need to train as
    # follows.
    K = np.unique(y_train)
    
    KC2 = np.ceil(len(K)*(len(K)-1)*0.5)

    # get the sklearn support vector classifier
    clf = SVC(kernel=kernel,C=C)
    
    ytest_pred = np.zeros((N,KC2))
    ytrain_pred = np.zeros((M, KC2))


    idx_pairs = []
    for i in range(len(K)):
        for j in range(i+1,len(K)):
            idx_pairs.append((K[i],K[j]))

    for idx, (ki, kj) in enumerate(idx_pairs):
        # if idx == KC2:
        #     break
        # if ki >= kj:
        #     continue

        #print idx, ki, kj

        # in sklearn's documentation y is {0,1} not {-1,1}
        # as in the lecture notes.
        ki_idx = (y_train == ki)
        kj_idx = (y_train == kj)

        Xtrain_ = np.vstack((X_train[ki_idx,:], X_train[kj_idx,:]))
        Ytrain_ = np.asarray(([1] * sum(ki_idx)) + ([0] * sum(kj_idx)))
        
        # train the classifier
        clf_k = clf.fit(Xtrain_, Ytrain_)
        
        # get the prediction (* k, ensures that {0,1} classification is changed to {0, k}) 
        ytrain_pred[:,idx] = clf_k.predict(X_train) * ki
        ytrain_pred[ytrain_pred[:,idx] == 0,idx] = kj

        ytest_pred[:,idx]  = clf_k.predict(X_test) * ki
        ytest_pred[ytest_pred[:,idx]==0,idx] = kj
        
    
    # # replace zeros with nan, helps when calculating mode
    # ytrain_pred[ytrain_pred[:,:] == 0] = np.nan
    # ytest_pred[ytest_pred[:,:] == 0] = np.nan
    # pdb.set_trace()

    # pick the most predicted class (apart from nan, done implicitly)
    ytrain_pred = stats.mode(ytrain_pred, axis=1)[0]
    ytest_pred  = stats.mode(ytest_pred, axis=1)[0] 

    assert ytrain_pred.shape == (N,1), "Count of training predictions must match the number of examples in training dataset "
    assert ytest_pred.shape == (M,1), "Count of test predictions must match the number of examples in test dataset "

    return (ytest_pred, ytrain_pred)


if __name__ == "__main__":
    
    # Load data
    X_train = np.loadtxt("./data/X_train.txt")
    X_test  = np.loadtxt("./data/X_test.txt")
    y_train = np.loadtxt("./data/y_train.txt")
    y_test  = np.loadtxt("./data/y_test.txt")

    [ytest_pred, ytrain_pred] = svmAllVsAll(X_train, X_test, y_train, kernel="linear", C=10)
    print(np.sum(ytest_pred == y_test.reshape(len(y_test),1)))
    print(np.sum(ytrain_pred == y_train.reshape(len(y_train),1)))
    pdb.set_trace()
