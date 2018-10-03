'''
Multiprocessing the function along the axis=0 by spliting the array into chucks
------------------------
Ziqiang Wei @ 2018
weiz@janelia.hhmi.org
'''

import multiprocessing as mp
import numpy as np


def parallel_to_chunks(func1d, arr, *args, **kwargs):
    if mp.cpu_count() == 1:
        raise ValueError('Multiprocessing is not running on single core cpu machines, and consider to change code.')

    mp_count = min(mp.cpu_count(), arr.shape[0]) # fix the error if arr is shorter than cpu counts
    print(f'Number of processes to parallel: {mp_count}')
    chunks = [(func1d, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, mp_count)]
    pool = mp.Pool(processes=mp_count)
    individual_results = pool.map(unpacking_apply_func, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    results = ()
    # print(len(individual_results[0]))
    for i_tuple in range(len(individual_results[0])):
        results = results + (np.concatenate([_[i_tuple] for _ in individual_results]), )
    return results


def unpacking_apply_func(list_params):
    func1d, arr, args, kwargs = list_params
    return func1d(arr, *args, **kwargs)


# this is a testing function
def print_shape(arr):
    print(arr.shape)
    return np.array([0]), np.array([1]),

def print_single_return(arr):
    print(arr.shape)
    return arr,

def test_():
    x = np.random.rand(100, 10, 2)
    arr0, arr1 = parallel_to_chunks(print_shape, x)
    print(arr0)

def test():
    x = np.random.rand(100, 10, 2)
    arr0 = parallel_to_chunks(print_single_return, x)
    print(arr0)

if __name__ == '__main__':
    test_()
