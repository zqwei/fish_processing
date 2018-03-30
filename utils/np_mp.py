import multiprocessing as mp
import numpy as np

# multiprocessing numpy along one dimension

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    if mp.cpu_count() == 1:
        raise ValueError('Multiprocessing is not running on single core cpu machines, and consider to change code.')

    print('%d cpus for multiprocessing'%(mp.cpu_count()))

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, mp.cpu_count())]

    pool = mp.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)

def unpacking_apply_along_axis(list_params):
    """
    Like numpy.apply_along_axis(), but and with arguments in a tuple
    instead.

    This function is useful with mp.Pool().map():
    (1) map() only handles functions that take a single argument, and
    (2) this function can generally be imported from a module, as required
    by map().
    """
    func1d, axis, arr, args, kwargs = list_params
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
