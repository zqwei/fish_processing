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


def get_local_cluster(dask_tmp=None, memory_limit='auto'):
    from dask.distributed import LocalCluster
    # from multiprocessing import cpu_count
    # n_workers = cpu_count()
    if dask_tmp is None:
        # return LocalCluster(n_workers=n_workers, processes=False, threads_per_worker=1, memory_limit=memory_limit)
        return LocalCluster(processes=False, memory_limit=memory_limit)
    else:
        # return LocalCluster(n_workers=n_workers, processes=False, threads_per_worker=1, local_dir=dask_tmp, memory_limit=memory_limit)
        return LocalCluster(processes=False, local_dir=dask_tmp, memory_limit=memory_limit) #n_workers=30, threads_per_worker=1, 


def get_jobqueue_cluster(walltime='12:00', ncpus=1, cores=1, local_directory=None, memory='15GB', env_extra=None, **kwargs):
    """
    Instantiate a dask_jobqueue cluster using the LSF scheduler on the Janelia Research Campus compute cluster.
    This function wraps the class dask_jobqueue.LSFCLuster and instantiates this class with some sensible defaults.
    Extra kwargs added to this function will be passed to LSFCluster().
    The full API for the LSFCluster object can be found here:
    https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.LSFCluster.html#dask_jobqueue.LSFCluster
    Some of the functions requires dask-jobqueue < 0.7
    """
    import dask
    # this is necessary to ensure that workers get the job script from stdin
    dask.config.set({"jobqueue.lsf.use-stdin": True})
    from dask_jobqueue import LSFCluster
    import os

    if env_extra is None:
        env_extra = [
            "export NUM_MKL_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export OPENMP_NUM_THREADS=1",
            "export OMP_NUM_THREADS=1",
        ]

    if local_directory is None:
        local_directory = '/scratch/' + os.environ['USER'] + '/'

    cluster = LSFCluster(queue='normal',
                         walltime=walltime,
                         ncpus=ncpus,
                         cores=cores,
                         local_directory=local_directory,
                         memory=memory,
                         env_extra=env_extra,
                         job_extra=["-o /dev/null"],
                         **kwargs)
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


def setup_workers(numCore=120, is_local=False, dask_tmp=None, memory_limit='auto'):
    from dask.distributed import Client
    if is_local:
        cluster = get_local_cluster(dask_tmp=dask_tmp, memory_limit=memory_limit)
        client = Client(cluster)
    else:
        cluster = get_jobqueue_cluster()
        # cluster.adapt(maximum_jobs=numCore)
        cluster.scale(numCore)
        client = Client(cluster)
        # cluster.start_workers(numCore)
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

        
def print_client_links(cluster):
    # print(f'Scheduler: {cluster.scheduler_address}')
    print(f'Dashboard link: {cluster.dashboard_link}')
    return None


if __name__ == '__main__':
    print('Testing setup of Dask---')
    cluster, client = setup_workers(10)
    print(client)
    print('Restart workers')
    client.restart()
    print(client)
    terminate_workers(cluster, client)
