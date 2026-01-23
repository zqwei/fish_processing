from fish_proc.wholeBrainDask.cellProcessing_single_WS import *
import dask.array as da
import numpy as np

df = pd.read_csv('data_list.csv')
dask_tmp = '/scratch/weiz/dask-worker-space'
memory_limit = 0 # unlimited
down_sample_registration = 3
baseline_percentile = 20
baseline_window = 1000   # number of frames
num_t_chunks = 2
# camera data
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'
# test data
dir_root = '/nearline/ahrens/Ziqiang_tmp/test_pipeline/'
# saving location
savetmp = '/scratch/weiz/'
save_root = savetmp

files = sorted(glob(dir_root+'/*.h5'))
chunks = File(files[0],'r')['default'].shape
nsplit = (chunks[1]//64, chunks[2]//64)
print('========================')
print('Preprocessing')
preprocessing(dir_root, savetmp, cameraNoiseMat=cameraNoiseMat, nsplit=nsplit, \
              num_t_chunks=num_t_chunks, dask_tmp=dask_tmp, memory_limit=memory_limit, \
              down_sample_registration=down_sample_registration, is_singlePlane=True)
print('========================')
print('Combining motion corrected data')
combine_preprocessing(dir_root, savetmp, num_t_chunks=num_t_chunks, dask_tmp=dask_tmp, memory_limit=memory_limit)

detrend_data(dir_root, savetmp, window=baseline_window, percentile=baseline_percentile, \
                 nsplit=nsplit, dask_tmp=dask_tmp, memory_limit=memory_limit)
print('========================')
print('Mask')
default_mask(dir_root, savetmp, dask_tmp=dask_tmp, memory_limit=memory_limit)
print('========================')
print('Demix')
dt = 3
is_skip = True
demix_cells(savetmp, dt, is_skip=is_skip, dask_tmp=dask_tmp, memory_limit=memory_limit)

# remove some files --
Y_d = da.from_zarr(f'{savetmp}/Y_max.zarr')
np.save(f'{save_root}/Y_max', Y_d.compute())
Y_d = da.from_zarr(f'{savetmp}/Y_d_max.zarr')
np.save(f'{save_root}/Y_d_max', Y_d.compute())
Y_d = da.from_zarr(f'{savetmp}/Y_ave.zarr')
chunks = Y_d.chunksize[:-1]
np.save(f'{save_root}/Y_ave', Y_d.compute())
np.save(f'{save_root}/chunks', chunks)

Y_d = np.load(f'{save_root}/Y_ave.npy')
chunks = np.load(f'{save_root}/chunks.npy')
Y_d_max = Y_d.max(axis=0, keepdims=True)
max_ = np.percentile(Y_d_max, 40)
mask_ = Y_d_max>max_
mask_ = np.repeat(mask_, Y_d.shape[0], axis=0)
mask_ = da.from_array(mask_, chunks=(1, chunks[1], chunks[2], -1))

print('========================')
print('DF/F computation')
compute_cell_dff_raw(savetmp, mask_, dask_tmp=dask_tmp, memory_limit=0)
combine_dff(savetmp)
combine_dff_sparse(savetmp)