'''
Dask processing
This is mainly a wrapper from Davis' code

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
    """
    Instatiate a DRMAACluster for use with the LSF scheduler on the Janelia Research Campus compute cluster. This is a
    wrapper for dask_drmaa.DRMMACluster that uses reasonable default settings for the dask workers. Specifically, this
    ensures that dask workers use the /scratch/$USER directory for temporary files and also that each worker runs on a
    single core. This wrapper also directs the $WORKER.err and $WORKER.log files to /scratch/$USER.
    """
    from dask_drmaa import DRMAACluster
    import os
    
    # we need these on each worker to prevent multithreaded numerical operations
    pre_exec =('export NUM_MKL_THREADS=1',
               'export OPENBLAS_NUM_THREADS=1',
               'export OPENMP_NUM_THREADS=1')
    local_directory = '/scratch/' + os.environ['USER']
    output_path = ':' + local_directory
    error_path = output_path
    cluster_kwargs_pass = {}
    cluster_kwargs_pass.setdefault(
        'template',
        {
            'args': [
                '--nthreads', '1',
                '--local-directory', local_directory],
            'jobEnvironment': os.environ,
            'outputPath': output_path,
            'errorPath': error_path,
        }
    )
    cluster_kwargs_pass['preexec_commands'] = pre_exec
    cluster = DRMAACluster(**cluster_kwargs_pass)
    return cluster


def setup_workers(numCore):
    cluster = setup_cluster()
    from dask.distributed import Client
    client = Client(cluster)
    cluster.start_workers(numCore)
    return cluster, client

def setup_local_worker():
    from dask.distributed import Client
    return Client()

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
