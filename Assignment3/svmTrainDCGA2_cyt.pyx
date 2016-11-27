#######################################################################
# Dual soft-margin SVM problem
# \text{max}_{a} J(a) = \sum_{i=1}^{N} a_i + 0.5 \sum_{i=1}^{N} \sum{j=1}^{N} a_i a_j y_i y_j k(x_i, x_j)
#
# s.t. 0 <= a_i <= C for all i = 1...N
#######################################################################
from __future__ import division
import numpy as np
cimport numpy as cnp
import pdb 

cpdef svmTrainDCGA(cnp.ndarray K, cnp.ndarray y, int C, int N=-1):
    """
    Solves the dual softmargin SVM using the 
    Stochastic Dual Coordinate Ascent algorithm.

    params K: kernel matrix (dim: NxN)
    params y: Label vectors (dim: Nx1)
    params C: Regularization parameter (dim: scalar)
    params N: Number of datapoints (dim: scalar)
    return a: Dual SVM variables (dim: Nx1)
    """

    # if not N:
    if N == -1:
        # usually the size of training data
        N = K.shape[1]

    cdef cnp.ndarray[cnp.float_t , ndim=2] a
    # Initialization
    a = np.zeros((N,1))
    cdef int count = 0 # number of iterations
    # threshold = np.exp(-3 * np.sqrt(np.trace(K)))
    cdef double threshold = np.exp(-3)* np.sqrt(np.trace(K))
    # threshold = np.clip(np.exp(-3 * np.sqrt(np.trace(K))), 10**-5, 0)

    cdef cnp.ndarray[cnp.float_t , ndim=2] grad
    grad = np.zeros_like(a)
    duality_gap = np.inf

    y_mat = np.expand_dims(y,axis=1)
    yK_full = np.repeat(y_mat, K.shape[1],axis=1) * K

    cdef int condition = True
    while condition:
        count += 1
        # Select a random training example (get a random index in 0:N-1)
        idx = np.random.randint(K.shape[1])
        
        # calculate the update direction delta_a
        delta_ai = 0 
        ##yK = a[:,0] * K[idx, :] 
        yK = y * K[idx, :] 
        # grad_i = 1 - y[idx] * np.sum(y * a * K[idx,:])
        grad_i = 1 - y[idx] * np.dot(a.T,yK)
        delta_ai = 1 if grad_i >= 0 else -1
        
        grad_i = np.clip(grad_i, -2, 2)
        # calculate step size t
        t = (delta_ai * grad_i)/K[idx,idx]
        if t != 0:
            ai_old = np.copy(a[idx])
            #a[idx] += np.clip((t * delta_ai), -ai_old, C-ai_old)
            a[idx] += t * delta_ai
            # clip a_i
            # a[idx] = np.minimum(C, np.maximum(a[idx],0))
            a[idx] = np.clip(a[idx],0,C)
            
            # update gradient
            # grad -= (y[idx]*(a[idx] - ai_old)*yK).reshape((grad.shape))
            grad = 1 - y * np.dot(a.T,yK_full) # TODO: Check with TA if this is fine.
            if count >= N:
                # grad = 1 - y * np.dot(a.T,yK)
                # pdb.set_trace()

                # # update gradient
                # grad = 1 - y * np.dot(a.T,yK_full)
                # grad = np.clip(grad, -2, 2)

                # calculate duality gap
                dg = []
                dg.append(np.dot(a.T,grad.T))
                dg.append(np.sum(np.maximum(grad, 0)))
                # duality_gap = -1 * np.dot(a.T,grad) + C * np.sum(np.maximum(grad, 0))# np.sum(grad[grad >=0]) more efficient ? 
                duality_gap = -1 * dg[0] + C * dg[1]# np.sum(grad[grad >=0]) more efficient ? 
                count = 0
                # since duality gap is only changing here
                # we can also update the condition to stop
                # the iteration here.
                condition = (duality_gap >= threshold)
                print("{} dg_1 = {} dg_2 = {} {} grad_max = {} grad_min = {} amax {} amin {}".format(duality_gap, dg[0], dg[1], threshold, np.max(grad), np.min(grad),np.max(a), np.min(a)))
    return a
