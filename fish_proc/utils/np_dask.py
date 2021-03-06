'''
Dask processing of numpy

Prerequisite:

Install dask-drmaa (server distributor)

conda install -c conda-forge dask-drmaa
------------------------
Ziqiang Wei @ 2018
weiz@janelia.hhmi.org
'''

import numpy as np
import warnings

def setup_cluster():
    from dask_drmaa import DRMAACluster
    import os
    cluster_kwargs_pass = {}
    cluster_kwargs_pass.setdefault(
           'template',
           {
               'args': [
               '--nthreads', '1',
               '--local-directory', '/scratch/' + os.environ['USER']],
               'jobEnvironment': os.environ
           }
       )
    cluster = DRMAACluster(**cluster_kwargs_pass)
    return cluster

def setup_workers(numCore):
    cluster = setup_cluster()
    from dask.distributed import Client
    client = Client(cluster)
    cluster.start_workers(numCore)
    return cluster, client

def terminate_workers(cluster, client):
    client.close()
    cluster.close()
    
def warn_on_large_chunks(x):
    import itertools
    shapes = list(itertools.product(*x.chunks))
    nbytes = [x.dtype.itemsize * np.prod(shape) for shape in shapes]
    if any(nb > 1e9 for nb in nbytes):
        warnings.warn("Array contains very large chunks")
        

if __name__ == '__main__':
    print('Testing setup of Dask---')
    cluster, client = setup_workers(10)
    print(client)
    print('Restart workers')
    client.restart()
    print(client)
    terminate_workers(cluster, client)
