#################################################
# Solving the dual soft margin SVM              #
# using quadratic programming solver            #  
# from a package (here we use scipy.optimize)   #
#################################################

from scipy.optimize import minimize
import numpy as np

def func(a,yyK=None,yK=None,y=None,sign=1):
    """
    Implements the dual softmargin svm equation (1)
    """
    print("inside func ..")
    accum = np.dot(np.dot(a.T, yyK),a)
    return sign*(np.sum(a) - 0.5 * accum).flatten() 

def func_deriv(a, yyK=None, yK=None, y=None,sign=1):
    """
    Implements the derivative of the dual softmargin svm.
    """
    print("calculating deriv..")
    retVal = sign*(1 - y.T * np.dot(a.T, yK)).T
    return retVal


def svmTrainQP(K, y, C):
    # initial estimate of alphas
    a0 = np.random.random((K.shape[1],1))
    
    y = np.expand_dims(y,axis=1)
    yyK = np.dot(y,y.T)*K
    print("Calculated yyK ..")

    y_rep = np.repeat(y,K.shape[1],axis=1)
    yK = y_rep * K
    print("Calculated yK ..")

    def callback(a):
        notin0C = np.invert((a >=0) * (a <= C))
        print("Callback ! {} not in [0,C] {}".format(func(a,yyK,yK,y),a[notin0C]))

    # constraint
    cons = ({'type':'ineq',
             'fun':lambda a: a,
             #'jac':lambda a: np.ones_like(a)},
             'jac':lambda a: a*0+1},
            {'type':'ineq',
             'fun':lambda a: C-a,
             'jac':lambda a: a*0-1})

    print(callback(a0))
    print(func_deriv(a0,yyK,yK,y))
    res = minimize(func, a0, args=(yyK,yK,y,-1), jac=func_deriv,callback=callback,constraints=cons)
    # res = minimize(func, a0, args=(yyK,yK,y), jac=False,options={'gtol': 1e-6, 'disp': True},callback=callback)
    print res
