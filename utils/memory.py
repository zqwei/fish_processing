'''
Memory monitor for large-scale array (memory leak/error tracker)
------------------------
Ziqiang Wei @ 2018
weiz@janelia.hhmi.org
'''

import time
import os
import psutil
import gc


def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        print("{}: memory before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before,
            elapsed_time))
        return result
    return wrapper

@profile
def list_create(n):
    print("inside a dummy list create")
    x = [1] * n
    return x

@profile
def clear_variables(x):
    if isinstance(x, tuple):
        for x_ in x:
            del x_
    else:
        del x_
    gc.collect()


def test():
    l = list_create(2000000)
    ll = list_create(2000)
    clear_variables((l,ll))

if __name__ == '__main__':
    test()
