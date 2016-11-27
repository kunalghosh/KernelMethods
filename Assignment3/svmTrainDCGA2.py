#######################################################################
# Dual soft-margin SVM problem
# $\text{max}_{a} J(a) = \sum_{i=1}^{N} a_i + 0.5 \sum_{i=1}^{N} \sum_{j=1}^{N} a_i a_j y_i y_j k(x_i, x_j)$
#
# s.t. $0 \leq a_i \leq C$ for all $i = 1\dots N$
#######################################################################
from __future__ import division
import numpy as np
import pdb 

def svmTrainDCGA(K, y, C, N=None):
    """
    Solves the dual softmargin SVM using the 
    Stochastic Dual Coordinate Ascent algorithm.

    params K: kernel matrix (dim: (N,N))
    params y: Label vectors (dim: (N,1))
    params C: Regularization parameter (dim: scalar)
    params N: Number of datapoints (dim: scalar)
    return a: Dual SVM variables (dim: (N,1))
    """

    if not N:
        # usually the size of training data
        N = K.shape[1]

    # Initialization
    a = np.zeros((N,1))
    count = 0 # number of iterations
    threshold = np.exp(-3)* np.sqrt(np.trace(K))

    grad = np.ones_like(a)
    duality_gap = np.inf

    y_mat = np.expand_dims(y,axis=1)
    yK_full = np.repeat(y_mat, K.shape[1],axis=1) * K

    condition = True
    while condition:
        count += 1
        # Select a random training example (get a random index in 0:N-1)
        idx = np.random.randint(K.shape[1])
        
        # calculate the update direction delta_a
        delta_ai = 0 
        yK = y * K[idx, :] 
        grad_i = 1 - y[idx] * np.dot(a.T,yK)
        delta_ai = 1 if grad_i >= 0 else -1
        
        # calculate step size t
        t = (1.0*delta_ai * grad_i)/K[idx,idx]

        if t != 0:
            ai_old = np.copy(a[idx])
           
            # update gradient
            t_del_a = np.clip(t*delta_ai, -ai_old, C-ai_old)
            grad -= (y[idx] * t_del_a * yK).reshape((grad.shape))

            a[idx] += t_del_a
            if count >= N:
                # # update gradient

                # calculate duality gap
                dg = []
                dg.append(np.dot(a.T,grad))
                dg.append(np.sum(np.maximum(grad, 0)))
                duality_gap = -1 * dg[0] + C * dg[1]
                count = 0
                # since duality gap is only changing here
                # we can also update the condition to stop
                # the iteration here.
                condition = (duality_gap >= threshold)
                # print("{} dg_1 = {} dg_2 = {} {} grad_max = {} grad_min = {} amax {} amin {}".format(duality_gap, dg[0], dg[1], threshold, np.max(grad), np.min(grad),np.max(a), np.min(a)))
    return a
