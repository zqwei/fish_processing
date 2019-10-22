#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

import os, sys
import warnings
warnings.filterwarnings('ignore')
from cellProcessing_single_WS import *
import dask.array as da
import numpy as np

df = pd.read_csv('data_list.csv')
dask_tmp = '/opt/data/weiz/dask-worker-space'
memory_limit = 0 # unlimited
down_sample_registration = 3

for ind, row in df.iterrows():
    if ind<3:
        continue
    dir_root = row['dat_dir']+'im/'
    save_root = row['save_dir']

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    files = sorted(glob(dir_root+'/*.h5'))
    chunks = File(files[0],'r')['default'].shape
    nsplit = (chunks[1]//64, chunks[2]//64)
    baseline_percentile = 20
    baseline_window = 1000   # number of frames
    num_t_chunks = 80
    cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    print('========================')
    print('Preprocessing')
    if not os.path.exists(f'{save_root}/motion_corrected_data.zarr'):
        preprocessing(dir_root, save_root, cameraNoiseMat=cameraNoiseMat, nsplit=nsplit, \
                      num_t_chunks=num_t_chunks, dask_tmp=dask_tmp, memory_limit=memory_limit, \
                      is_bz2=False, down_sample_registration=down_sample_registration)

    if not os.path.exists(f'{save_root}/detrend_data.zarr'):
        detrend_data(dir_root, save_root, window=baseline_window, percentile=baseline_percentile, \
                     nsplit=nsplit, dask_tmp=dask_tmp, memory_limit=memory_limit)


    print('========================')
    print('Mask')
    default_mask(dir_root, save_root, dask_tmp=dask_tmp, memory_limit=memory_limit)


    print('========================')
    print('Demix')
    dt = 3
    is_skip = True
    demix_cells(save_root, dt, is_skip=is_skip, dask_tmp=dask_tmp, memory_limit=memory_limit)
    
