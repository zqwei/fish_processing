import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pdastf.pdas.trendfilter import TF, TFsafe, TFsafeG
from itertools import product


def plot(x,collector,fname = 'test_randomtf.png'):
    'Generate plot on tested results'
    plt.figure(figsize = (6,6))
    plt.title('Trend of indicators')
    plt.subplot(2,1,1)
    plt.plot(x,collector['iterations']['pdas'],'ro-',label='pdas',ms=3)
    plt.plot(x,collector['iterations']['safe1'],'bo-',label='safe-1',ms=3)
    plt.plot(x,collector['iterations']['safe2'],'go-',label='safe-2',ms=3)
    plt.plot(x,collector['iterations']['safe3'],'mo-',label='safe-3',ms=3)
    plt.ylabel('Iteration')
    plt.xlabel('Size')
    plt.legend(prop={'size':6},loc=2)

    plt.subplot(2,1,2)
    plt.plot(x,collector['times']['pdas'],'ro-',label='pdas',ms=3)
    plt.plot(x,collector['times']['safe1'],'bo-',label='safe-1',ms=3)
    plt.plot(x,collector['times']['safe2'],'go-',label='safe-2',ms=3)
    plt.plot(x,collector['times']['safe3'],'mo-',label='safe-3',ms=3)
    plt.ylabel('Time')
    plt.xlabel('Size')
    plt.legend(prop={'size':6},loc=2)
    plt.savefig(fname)


if __name__=='__main__':
    
    size = 36

    seed_partition = 0
    seed_data = 0

    np.random.seed(seed_data)


    for order, mode in product((2,1),(-1,0)):
        iterations = {'pdas':[],'safe1':[],'safe2':[],'safe3':[]}
        times = {'pdas':[],'safe1':[],'safe2':[],'safe3':[]}


        for i in range(1,size):
            # Generate data
            dat = np.arange(i * 1e4).astype(np.float)
            dat += 2 * np.random.randn(i * 1e4)  # add noise

            # By pdas
            print('--- by pdas --')
            random.seed(seed_partition)
            prob = TF(dat[:,np.newaxis],1000,order=order,mode=mode)
            prob.pdas()
            if prob.info['status'] == 'optimal':
                iterations['pdas'].append(prob.info['iter'])
                times['pdas'].append(prob.info['time'])
            else:
                iterations['pdas'].append(0)
                times['pdas'].append(0)
            del prob

            # By pdas with safeguard
            print('--- by pdas with safeguard --')
            random.seed(seed_partition)
            prob = TFsafe(dat[:,np.newaxis],1000,order=order,mode=mode)
            prob.pdas()
            if prob.info['status'] == 'optimal':
                iterations['pdas'].append(prob.info['iter'])
                times['pdas'].append(prob.info['time'])
            else:
                iterations['pdas'].append(0)
                times['pdas'].append(0)
            del prob

            # By pdas with safeguard 2
            print('--- by pdas with safeguard 2 --')
            random.seed(seed_partition)
            prob = TFsafe(dat[:,np.newaxis],1000,order=order,mode=mode)
            prob.pdas2()
            if prob.info['status'] == 'optimal':
                iterations['pdas'].append(prob.info['iter'])
                times['pdas'].append(prob.info['time'])
            else:
                iterations['pdas'].append(0)
                times['pdas'].append(0)
            del prob

            # By pdas with safeguard 3
            print('--- by pdas with safeguard 3 roll back --')
            random.seed(seed_partition)
            prob = TFsafeG(dat[:,np.newaxis],1000,order=order,mode=mode)
            prob.pdas2()
            if prob.info['status'] == 'optimal':
                iterations['pdas'].append(prob.info['iter'])
                times['pdas'].append(prob.info['time'])
            else:
                iterations['pdas'].append(0)
                times['pdas'].append(0)
            del prob

        collector = {'iterations':iterations, 'times':times}
        plot([i for i in range(1,size)],collector,fname = 'order'+str(order)+'mode'+str(mode)+'.pdf')
