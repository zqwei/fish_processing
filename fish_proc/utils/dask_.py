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
    if dask_tmp is None:
        return LocalCluster(processes=False, memory_limit=memory_limit)
    else:
        return LocalCluster(processes=False, local_dir=dask_tmp, memory_limit=memory_limit)


def init_cluster(num_workers, wait_for_all_workers=True):
    """
    Start up a dask cluster, optionally wait until all workers have been launched,
    and then return the resulting distributed.Client object.

    Args:
        num_workers:
            How many workers to launch.
        wait_for_all_workers:
            If True, pause until all workers have been launched before returning.
            Otherwise, just wait for a single worker to launch.

    Returns:
        distributed.Client
    """
    # Local import: LSFCluster probably isn't importable on your local machine,
    # so it's nice to avoid importing it when you're just running local tests without a cluster.
    from dask_jobqueue import LSFCluster
    from distributed import Client
    import time
    cluster = LSFCluster(ip='0.0.0.0')
    cluster.scale(num_workers)

    required_workers = 1
    if wait_for_all_workers:
        required_workers = num_workers

    client = Client(cluster)
    while (wait_for_all_workers and
           client.status == "running" and
           len(cluster.scheduler.workers) < required_workers):
        print(f"Waiting for {required_workers - len(cluster.scheduler.workers)} workers...")
        time.sleep(1.0)

    return cluster, client

def get_jobqueue_cluster(num_workers):
    import dask
    import os
    dask.config.set({'jobqueue':
                        {'lsf':
                          {'cores': 1,
                           'memory': '15GB',
                           'walltime': '01:00',
                           'log-directory': 'dask-logs',
                           'local-directory': f'/scratch/{os.environ["USER"]}',
                           'use-stdin': True}
                           }
                           })

    return init_cluster(num_workers)


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
        cluster, client = get_jobqueue_cluster(numCore)
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
