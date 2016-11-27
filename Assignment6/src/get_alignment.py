import numpy as np
from cvxopt import matrix
from cvxopt import solvers

def align(K1, K2):
    # return np.trace(np.dot(K1,K2))
    return np.einsum('ij,ji->',K1,K2)

def populate_C(args, Ky):
    c = np.zeros((len(args),1))
    for idx,Ki in enumerate(args):
        c[idx] = align(Ki,Ky)
    return c

def populate_Q(args):
    T = len(args) # Number of kernel matrices
    Q = np.zeros((T,T))
    idxs = range(T)

    for row in idxs:
        for col in idxs[row:]:
            Q[row,col] = align(args[row], args[col])
            Q[col,row] = Q[row,col]
    return Q

def center_align(Kernels):
    m = Kernels[0].shape[0]
    center_matrix = np.eye(m) - np.ones_like(Kernels[0]) / m

    for idx,kernel in enumerate(Kernels):
        Kernels[idx] = np.dot(np.dot(center_matrix, kernel),center_matrix)

    return Kernels
         

def get_alignment(Kernels, Y):
    """
    This function accepts a set of one or more
    Kernel Matrices and returns a vector 'd' 
    which can be used to calculate the weighted sum
    of these Kernel Matrices such that the sum minimizes
    the alignment between input kernels and the target
    given by Y.
    Y is an (N,K) matrix where N = Number of training and test
    data points. K = Number of functional categories. 

    param Kernels : Arbitrary number (T) of kernel matrices. (N,N) Matrices
    param       Y : Target values. (N,K)  
    return      d : A vector of weights. (T,1)
    """
    # center align the matrices
    args = center_align(Kernels)
    T = len(args)

    Ky = np.dot(Y,Y.T)
    c = populate_C(args, Ky)
    Q = populate_Q(args)

    # Solve the quadratic program
    P = matrix(Q)
    q = matrix(-1 * c)
    G = matrix(-1 * np.eye(T))
    h = matrix(np.zeros((T,1)))

    sol = solvers.qp(P,q,G,h)
    d = np.asarray(sol['x'])

    return d / np.linalg.norm(d)

