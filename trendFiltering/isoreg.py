# Author  : Zheng Han <zhh210@lehigh.edu>
# About   : isotonic regression with primal-dual active-set (PDAS) method
# License : BSD

import numpy as np
from skiplistcollections import SkipListSet
from itertools import chain, count, groupby
from collections import OrderedDict
from blist import blist

import pdb
def isotonic_regression(y, sample_weight=None, y_min=None, y_max=None,
                        increasing=True):
    """Interface to solve the isotonic regression model::
        min sum w[i] (y[i] - y_[i]) ** 2
        subject to y_min = y_[1] <= y_[2] ... <= y_[n] = y_max
    where:
        - y[i] are inputs (real numbers)
        - y_[i] are fitted
        - w[i] are optional strictly positive weights (default to 1.0)
    Parameters
    ----------
    y : iterable of floating-point values
        The data.
    sample_weight : iterable of floating-point values, optional, default: None
        Weights on each point of the regression.
        If None, weight is set to 1 (equal weights).
    y_min : optional, default: None
        If not None, set the lowest value of the fit to y_min.
    y_max : optional, default: None
        If not None, set the highest value of the fit to y_max.
    increasing : boolean, optional, default: True
        Whether to compute ``y_`` is increasing (if set to True) or decreasing
        (if set to False)
    Returns
    -------
    y_ : list of floating-point values
        Isotonic fit of y.
    References
    ----------
    "PDAS on isotonic regression"
    """
    y = np.asarray(y, dtype=np.float)
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=y.dtype)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float)
    if not increasing:
        y = y[::-1]
        sample_weight = sample_weight[::-1]

    if y_min is not None or y_max is not None:
        y = np.copy(y)
        sample_weight = np.copy(sample_weight)
        # upper bound on the cost function, will force y_max or y_min reached
        C = np.dot(sample_weight, y * y) * 10
        if y_min is not None:
            y[0] = y_min
            sample_weight[0] = C
        if y_max is not None:
            y[-1] = y_max
            sample_weight[-1] = C

    y_ = _isotonic_regression2(y, sample_weight)
    if increasing:
        return y_
    else:
        return y_[::-1]

def _isotonic_regression(y,sample_weight,blocks = None):
    # Without code 'optimization, plain pdas, depreciated'
    """Solve the isotonic regression model::
        min sum w[i] (y[i] - y_[i]) ** 2
        subject to y_[1] <= y_[2] ... <= y_[n]
    where:
        - y[i] are inputs (real numbers)
        - y_[i] are fitted
        - w[i] are optional strictly positive weights (default to 1.0)
    Parameters
    ----------
    y : iterable of floating-point values
        The data.
    sample_weight : iterable of floating-point values, optional, default: None
        Weights on each point of the regression.
        If None, weight is set to 1 (equal weights).
    Returns
    -------
    y_ : list of floating-point values
        Isotonic fit of y.
    """
    # Length of data
    n = len(sample_weight)
    w = sample_weight
    # Initialize solution
    y_ = np.empty(n,dtype = float)
    z_ = np.empty(n-1,dtype = float)
    # Guess a block
    if blocks is None:
        blocks = SkipListSet(capacity=n+5)
        for i in range(1,n+1,1): blocks.add(i)
    else:
        blocks.add(n)
        

    # Main loop
    while True:
        # Compute solution for each block
        vp = []
        vd = []
        lb = list(blocks)
        positions = chain([(0,lb[0])],[(lb[i-1],lb[i]) for i in range(1,len(lb),1)])

        for istart, iend in positions:
            # Compute primal
            y_[istart:iend] = np.ones(iend-istart)*np.dot(y[istart:iend],w[istart:iend])/np.sum(w[istart:iend])
            # Compute dual
            # if istart != 0:
            #     z_[istart-1] = 0
            # z_[istart:iend-1] = w[istart:iend-1]*(y[istart:iend-1] - y_[istart:iend-1])
            # for i in range(istart+1,iend-1):
            #     z_[i] += z_[i-1]
        # Obtain violations
        lb = list(blocks)
        positions = chain([(0,lb[0])],[(lb[i-1],lb[i]) for i in range(1,len(lb),1)])
        for istart, iend in positions:
            if iend < n and y_[iend-1] > y_[iend]:
                vp.append(iend)

        # for i in np.where(z_ < 0)[0]:
        #     vd.append(i+1)


        for i in vp: blocks.remove(i)
#        for j in vd: blocks.add(j)


        # Check optimality
        if len(vp) + len(vd) == 0:
            break

    return y_
            


def _isotonic_regression2(y,sample_weight,blocks = None):
    'PDAS customized for isotonic regression'
    """Solve the isotonic regression model::
        min sum w[i] (y[i] - y_[i]) ** 2
        subject to y_[1] <= y_[2] ... <= y_[n]
    where:
        - y[i] are inputs (real numbers)
        - y_[i] are fitted
        - w[i] are optional strictly positive weights (default to 1.0)
    Parameters
    ----------
    y : iterable of floating-point values
        The data.
    sample_weight : iterable of floating-point values, optional, default: None
        Weights on each point of the regression.
        If None, weight is set to 1 (equal weights).
    Returns
    -------
    y_ : list of floating-point values
        Isotonic fit of y.
    """
    # Length of data
    n = len(sample_weight)
    w = sample_weight
    wy = sample_weight*y
    # Initialize solution
    y_ = np.empty(n,dtype = float)
    z_ = np.empty(n-1,dtype = float)

    # Initialize blocks
    # blocks = dict(zip(range(n),[{'end':i, 'sum_w':w[i], 'sum_wy':wy[i], 'val':y[i]} for i in range(n)]))
    blocks = blist([{'begin':i, 'end':i+1, 'sum_w':w[i], 'sum_wy':wy[i], 'val':y[i]} for i in range(n)])

    # Main loop
    while True:
        violations = [i for i in range(len(blocks)-1) if blocks[i]['val'] > blocks[i+1]['val']]

        if len(violations) == 0: break

        groupedv = [list(g) for _, g in groupby(violations, lambda n, c=count(): n-next(c))]

        # Merge blocks
        offset = 0
        for block in groupedv:
            vblock = block+[block[-1]+1]
            temp = [blocks[i - offset] for i in vblock]
            new_item = {'begin': temp[0]['begin'], 'end': temp[-1]['end']}
            new_item['sum_w'] = sum([p['sum_w'] for p in temp])
            new_item['sum_wy'] = sum([p['sum_wy'] for p in temp])
            new_item['val'] = new_item['sum_wy']/new_item['sum_w']
            blocks[vblock[0]-offset] = new_item

            for i in vblock[1:]:
                blocks.pop(i - offset)
                offset += 1

    # Post process data
    for block in blocks:
        for i in range(block['begin'],block['end']): y_[i] = block['val']
    
    return y_
