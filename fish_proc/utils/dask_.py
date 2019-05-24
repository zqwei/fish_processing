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


def get_jobqueue_cluster(walltime='12:00', cores=1, local_directory=None, memory='16GB', **kwargs):
    """
    Instantiate a dask_jobqueue cluster using the LSF scheduler on the Janelia Research Campus compute cluster.
    This function wraps the class dask_jobqueue.LSFCLuster and instantiates this class with some sensible defaults.
    Extra kwargs added to this function will be passed to LSFCluster().
    The full API for the LSFCluster object can be found here:
    https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.LSFCluster.html#dask_jobqueue.LSFCluster

    """
    from dask_jobqueue import LSFCluster
    import os

    if local_directory is None:
        local_directory = '/scratch/' + os.environ['USER'] + '/'

    cluster = LSFCluster(queue='normal',
                         walltime=walltime,
                         cores=cores,
                         local_directory=local_directory,
                         memory=memory,
                         **kwargs)
    return cluster


def get_local_cluster(dask_tmp=None, memory_limit='auto'):
    from dask.distributed import LocalCluster
    if dask_tmp is None:
        return LocalCluster(processes=False, memory_limit=memory_limit)
    else:
        return LocalCluster(processes=False, local_dir=dask_tmp, memory_limit=memory_limit)


def setup_drmma_cluster():
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


def setup_cluster_workers(numCore):
    cluster = get_jobqueue_cluster()
    from dask.distributed import Client
    client = Client(cluster)
    cluster.start_workers(numCore)
    return cluster, client


def setup_local_worker():
    from dask.distributed import Client
    return Client()


def setup_workers(numCore=70, is_local=False, dask_tmp=None, memory_limit='auto'):
    from dask.distributed import Client
    if is_local:
        cluster = get_local_cluster(dask_tmp=dask_tmp, memory_limit=memory_limit)
        client = Client(cluster)
    else:
        cluster = get_jobqueue_cluster()
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
