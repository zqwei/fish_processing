import numpy as np
from pdastf.pdas.isoreg import isotonic_regression as pdas_iso 
import pdb
from sklearn.isotonic import IsotonicRegression
from sklearn.isotonic import isotonic_regression as iso_sklearn_fun
from ipm import rfilter
# Author : Fabian Pedregosa <fabian@fseoane.net>
# Adapted: Zheng Han <zhh210@lehigh.edu>
# License : BSD
 
 
def isotonic_regression_new(w, y, x_min=None, x_max=None):
    """
    Solve the isotonic regression with complete ordering model:

        min_x Sum_i{ w_i (y_i - x_i) ** 2 }

        subject to x_min = x_1 <= x_2 ... <= x_n = x_max

    where each w_i is strictly positive and each y_i is an arbitrary
    real number.

    Parameters
    ----------
    w : iterable of floating-point values
    y : iterable of floating-point values

    Returns
    -------
    x : list of floating-point values
    """
 
    if x_min is not None or x_max is not None:
        y = np.copy(y)
        w = np.copy(w)
        C = np.dot(w, y * y) * 10 # upper bound on the cost function
        if x_min is not None:
            y[0] = x_min
            w[0] = C
        if x_max is not None:
            y[-1] = x_max
            w[-1] = C
 
    J = [(w[i] * y[i], w[i], [i,]) for i in range(len(y))]
    cur = 0
 
    while cur < len(J) - 1:
        v0, v1, v2 = 0, 0, np.inf
        w0, w1, w2 = 1, 1, 1
        while v0 * w1 <= v1 * w0 and cur < len(J) - 1:
            v0, w0, idx0 = J[cur]
            v1, w1, idx1 = J[cur + 1]
            if v0 * w1 <= v1 * w0:
                cur +=1
 
        if cur == len(J) - 1:
            break
 
        # merge two groups
        v0, w0, idx0 = J.pop(cur)
        v1, w1, idx1 = J.pop(cur)
        J.insert(cur, (v0 + v1, w0 + w1, idx0 + idx1))
        while v2 * w0 > v0 * w2 and cur > 0:
            v0, w0, idx0 = J[cur]
            v2, w2, idx2 = J[cur - 1]
            if w0 * v2 >= w2 * v0:
                J.pop(cur)
                J[cur - 1] = (v0 + v2, w0 + w2, idx0 + idx2)
                cur -= 1
 
    sol = np.empty(len(y))
    for v, w, idx in J:
        sol[idx] = v / w
    return sol
 
 
 
def isotonic_regression(y, weights=[]):
    # Expensive violation checking, overhead of np.where(np.diff(v))
    # No pre-computation of weights[id]*y[id]
    """
    Solve the isotonic regression model:

        min Sum{ weights_i (x_i - y_i) ** 2 }

        subject to x_0 <= x_1 < ... < x_n

    Parameters
    ----------
    y : array-like, shape=(n_samples,)
        Input data.

    w : array-like, shape=(n_samples,), optional
        Weights in the cost function (must be strictly
        positive numbers),

    Returns
    -------
    x : array
    """
    y, weights = map(np.asarray, (y, weights))
    assert y.ndim == 1
    if weights.size:
        assert weights.ndim == 1
        assert weights.size == y.size
    n_samples = len(y)
    v = y.copy()
    lvls = np.arange(n_samples)
    lvlsets = np.c_[lvls, lvls]
    viol = np.where(np.diff(v) < 0)[0]
    while viol.size:
        start = lvlsets[viol[0], 0]
        last = lvlsets[viol[0] + 1, 1]
 
        n = last - start + 1
        idx = slice(start, last + 1)
        if weights.size:
            v[idx] = np.dot(weights[idx], y[idx]) / np.sum(weights[idx])
        else:
            v[idx]= np.sum(v[idx]) / n
        lvlsets[idx, 0] = start
        lvlsets[idx, 1] = last
        viol = np.where(np.diff(v) < 0)[0]
    return v
 
 
def isotonic_regression_2(w, y, x_min=None, x_max=None):
    # No pre-computation of weights[id]*y[id]
    """
    Solve the isotonic regression model:

        min Sum w_i (y_i - x_i) ** 2

        subject to x_min = x_1 <= x_2 ... <= x_n = x_max

    where each w_i is strictly positive and each y_i is an arbitrary
    real number.

    Parameters
    ----------
    w : iterable of floating-point values
    y : iterable of floating-point values

    Returns
    -------
    x : list of floating-point values
    """
 
    if x_min is not None or x_max is not None:
        y = np.copy(y)
        w = np.copy(w)
        if x_min is not None:
            y[0] = x_min
            w[0] = 1e32
        if x_max is not None:
            y[-1] = x_max
            w[-1] = 1e32
 
    J = [[i,] for i in range(len(y))]
    cur = 0
 
    while cur < len(J) - 1:
        av0, av1, av2 = 0, 0, np.inf
        while av0 <= av1 and cur < len(J) - 1:
            idx0 = J[cur]
            idx1 = J[cur + 1]
            av0 = np.dot(w[idx0], y[idx0]) / np.sum(w[idx0])
            av1 = np.dot(w[idx1], y[idx1]) / np.sum(w[idx1])
            cur += 1 if av0 <= av1 else 0
 
        if cur == len(J) - 1:
            break
 
        a = J.pop(cur)
        b = J.pop(cur)
        J.insert(cur, a + b)
        while av2 > av0 and cur > 0:
            idx0 = J[cur]
            idx2 = J[cur - 1]
            av0 = np.dot(w[idx0], y[idx0]) / np.sum(w[idx0])
            av2 = np.dot(w[idx2], y[idx2]) / np.sum(w[idx2])
            if av2 >= av0:
                a = J.pop(cur - 1)
                b = J.pop(cur - 1)
                J.insert(cur - 1, a + b)
                cur -= 1
 
    sol = []
    for idx in J:
        sol += [np.dot(w[idx], y[idx]) / np.sum(w[idx])] * len(idx)
    return np.asarray(sol)
 
 
if __name__ == '__main__':
    from time import time
    np.random.seed(0)
    t1, t2, t3, t4, t5, t6 = [], [], [], [], [], []
    size = 36
    for i in range(1, size):
        print(i)
        dat = np.arange(i * 1e4).astype(np.float)
        dat += 2 * np.random.randn(i * 1e4)  # add noise
        weights = .5 + np.abs(np.random.randn(i * 1e4))

        # sklearn, C version
        start = time()
        dat_hat = iso_sklearn_fun(dat,sample_weight = weights)        
 
        t1.append(time() - start)
        # pava_1
        # start = time()
        # dat_hat2 = isotonic_regression_2(weights, dat)
        # t2.append(time() - start)
        # print 'Result matches: ', np.allclose(dat_hat, dat_hat2)
 
        # pava_2, optimized
        start = time()
        dat_hat2 = isotonic_regression_new(weights, dat)
        t3.append(time() - start)
        print 'Result matches: ', np.allclose(dat_hat, dat_hat2)

        # pdas
        start = time()
        dat_hat2 = pdas_iso(dat,weights)
        t4.append(time() - start)
        print 'Result matches: ', np.allclose(dat_hat, dat_hat2)

        # ipm
        start = time()
        prob = rfilter(weights,dat,vlam = 100,reg = 'lm', order = 1)
        dat_hat2 = np.array(prob.variables()[0].value).reshape((max(dat_hat2.shape),))
        t5.append(time() - start)
        print 'Result matches: ', np.allclose(dat_hat, dat_hat2)

        # sklearn, C version
        start = time()
        ir = IsotonicRegression()
        dat_hat2 = iso_sklearn_fun(dat,sample_weight = weights)        
        t6.append(time() - start)
        print 'Result matches: ', np.allclose(dat_hat, dat_hat2)
        
    import pylab as pl
    n_samples = [i * 1e4 for i in range(size -1)]
    pl.plot(n_samples, t3, color='green', label='PAVA')
    pl.plot(n_samples, t5, color='blue', label='IPM')
    pl.plot(n_samples, t4, color='red', label='PDAS')
    # pl.plot(n_samples, t6, color='yellow', label='SCIKIT')
    print t1, t2, t3, t4, t5, t6
    pl.xlabel('Number of Samples')
    pl.ylabel('Time Elapse (s)')
    pl.legend(loc = 'upper left')
    pl.savefig('isotonic_compare.png')
    pl.show()
