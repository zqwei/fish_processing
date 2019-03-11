'''
Dask processing
This is mainly an example from Sturat's code

Prerequisite:

Install dask-jobqueue

conda install -c conda-forge dask-jobqueue
------------------------
Salma Elmalaki @ 2018
elmalakis@janelia.hhmi.org
'''

import time
import dask.bag as db
from distributed import Client

def init_cluster(num_workers, wait_for_all_workers=True):

    from dask_jobqueue import LSFCluster  #Unresolved import

    cluster = LSFCluster(ip='0.0.0.0')
    cluster.scale(num_workers)
    client = Client(cluster)

    while (wait_for_all_workers and
           client.status == "running" and
           len(cluster.scheduler.workers) < num_workers):
        time.sleep(1.0)

    return client


# Test
# bsub -n 1 -J test -Is /bin/bash
# export DASK_CONFIG=dask-config.yaml
# Then run the code
# client = init_cluster(2)
# try:
#     def double(x):
#         return 2*x
#
#
#     bag = db.from_sequence(range(100))
#     doubled = bag.map(double).compute()
#     print(doubled)
#
#
# finally:
#     client.close()
#     client.cluster.close()