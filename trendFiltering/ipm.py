import cvxpy as cvx
from cvxopt import spmatrix, spdiag
import numpy as np

def rfilter(weights,y,vlam = 10,reg = 'l1', order = 1):
    '''
    Approximate y with a:
     (reg, order) = ('l1',1) - piecewise constant-shaped function
     (reg, order) = ('l1',2) - piecewise linear-shaped function
     (reg, order) = ('lm',1) - (nearly) isotonic regression
     (reg, order) = ('lm',2) - (nearly) convex approximation
    '''
    # Prepare elements common for all models
    n = max(y.shape)
    x = cvx.Variable(n)
    lam = cvx.Parameter(sign = 'positive')
    lam.value = vlam

    # Prepare difference operator matrix for abs regularization
    if order == 1:
        D = spmatrix([],[],[],size = (n-1,n))
        for i in range(n-1):
            D[i,i] = 1
            D[i,i+1] = -1
    elif order == 2:
        D = spmatrix([],[],[],size = (n-2,n))
        for i in range(n-2):
            D[i,i] = -1
            D[i,i+1] = 2
            D[i,i+2] = -1
        
    # Prepare objective function
    res = cvx.vec(weights).T*cvx.square(y-x)
    #res = weights.T*cvx.square(y - x)
    if reg == 'l1':
        # l-1 regularization
        lreg = lam*cvx.sum_entries(cvx.abs(D*x))
    elif reg == 'lm':
        # unilaterally penalized
        lreg = lam*cvx.sum_entries(cvx.pos(D*x))

    obj = cvx.Minimize(res + lreg)
    prob = cvx.Problem(obj)
    prob.solve()
    return prob

if __name__ == '__main__':
    np.random.seed(1)
    y = np.random.randn(100,1)
    weights = .5 + np.abs(np.random.randn(100))
    vlam = 1
    prob = rfilter(weights,y,vlam,order = 1, reg = 'lm')
    print prob.value
