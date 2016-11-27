#################################################
# Solving the dual soft margin SVM              #
# using quadratic programming solver            #  
# from a package (CVXOPT)                       #
#################################################

import pdb
import numpy as np
from cvxopt import matrix
from cvxopt import solvers


def svmTrainQP(K, y, C):
    """
    params K: (N_train,N_train) kernel matrix.
    params y: (N_train,) training label vector.
    params C: Scalar

    solvers.qp solves a quadratic program
    of the form:
    min 0.5 x' P x + q' x 
     x
    
    s.t. Gx <= h

    where x' implies x tranpose
    """
    y = np.expand_dims(y, axis=1)
    N = K.shape[0]

    P = matrix(np.dot(y,y.T) * K, tc='d')
    q = matrix(-1*np.ones_like(y), tc='d')
    G = matrix(np.concatenate((
                np.eye(N)*-1,
                np.eye(N)
                )))
    h = matrix(np.concatenate((
                np.zeros_like(y),
                np.ones_like(y)*C
                )))

    sol = solvers.qp(P,q,G,h)
    return sol['x']
