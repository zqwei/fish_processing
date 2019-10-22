import numpy as np
import pandas as pd
import os, sys, gc, shutil, time
from glob import glob
from h5py import File
import warnings
warnings.filterwarnings('ignore')
import dask.array as da
from utils import *
import fish_proc.utils.dask_ as fdask
from fish_proc.utils.getCameraInfo import getCameraInfo
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'


def print_client_links(cluster):
    print(f'Scheduler: {cluster.scheduler_address}')
    print(f'Dashboard link: {cluster.dashboard_link}')
    return None


def preprocessing(dir_root, save_root, cameraNoiseMat=cameraNoiseMat, nsplit = (4, 4), num_t_chunks = 80,\
                  dask_tmp=None, memory_limit=0, is_bz2=False, is_singlePlane=False, down_sample_registration=1):
    # set worker
    cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print_client_links(cluster)

    if not os.path.exists(f'{save_root}/denoised_data.zarr'):
        if not is_bz2:
            files = sorted(glob(dir_root+'/*.h5'))
            chunks = File(files[0],'r')['default'].shape
            if not is_singlePlane: 
                data = da.stack([da.from_array(File(fn,'r')['default'], chunks=chunks) for fn in files])
            else:
                if len(chunks)==2:
                    data = da.stack([da.from_array(File(fn,'r')['default'], chunks=chunks) for fn in files])
                else:
                    data = da.concatenate([da.from_array(File(fn,'r')['default'], chunks=(1, chunks[1], chunks[2])) for fn in files], axis=0)
            cameraInfo = getCameraInfo(dir_root)
        else:
            import dask
            import xml.etree.ElementTree as ET
            from utils import load_bz2file
            dims = ET.parse(dir_root+'/ch0.xml')
            root = dims.getroot()
            for info in root.findall('info'):
                if info.get('dimensions'):
                    dims = info.get('dimensions')
            dims = dims.split('x')
            dims = [int(float(num)) for num in dims]
            files = sorted(glob(dir_root+'/*.stack.bz2'))
            imread = dask.delayed(lambda v: load_bz2file(v, dims), pure=True)
            lazy_data = [imread(fn) for fn in files]
            sample = lazy_data[0].compute()
            data = da.stack([da.from_delayed(fn, shape=sample.shape, dtype=sample.dtype) for fn in lazy_data])
            cameraInfo = getCameraInfo(dir_root)
            pixel_x0, pixel_x1, pixel_y0, pixel_y1 = [int(_) for _ in cameraInfo['camera_roi'].split('_')]
            pixel_x0 = pixel_x0-1
            pixel_y0 = pixel_y0-1
            cameraInfo['camera_roi'] = '%d_%d_%d_%d'%(pixel_x0, pixel_x1, pixel_y0, pixel_y1)
            chunks = sample.shape
        # pixel denoise
        if not is_singlePlane: 
            denoised_data = data.map_blocks(lambda v: pixelDenoiseImag(v, cameraNoiseMat=cameraNoiseMat, cameraInfo=cameraInfo))
        else:
            denoised_data = data.map_blocks(lambda v: pixelDenoiseImag(v, cameraNoiseMat=cameraNoiseMat, cameraInfo=cameraInfo), new_axis=1)
        denoised_data.to_zarr(f'{save_root}/denoised_data.zarr')
        num_t = denoised_data.shape[0]
    else:
        denoised_data = da.from_zarr(f'{save_root}/denoised_data.zarr')
        chunks = denoised_data.shape[1:]
        num_t = denoised_data.shape[0]

    # save and compute reference image
    print('Compute reference image ---')
    if not os.path.exists(f'{save_root}/motion_fix_.h5'):
        med_win = len(denoised_data)//2
        ref_img = denoised_data[med_win-50:med_win+50].mean(axis=0).compute()
        save_h5(f'{save_root}/motion_fix_.h5', ref_img, dtype='float16')

    print('--- Done computing reference image')

    # compute affine transform
    print('Registration to reference image ---')
    # create trans_affs file
    if not os.path.exists(f'{save_root}/trans_affs.npy'):
        ref_img = File(f'{save_root}/motion_fix_.h5', 'r')['default'].value
        ref_img = ref_img.max(axis=0, keepdims=True)
        if down_sample_registration==1:
            trans_affine = denoised_data.map_blocks(lambda x: estimate_rigid2d(x, fixed=ref_img), dtype='float32', drop_axis=(3), chunks=(1,4,4)).compute()
        else:
            #### downsample trans_affine case
            trans_affine = denoised_data[0::down_sample_registration].map_blocks(lambda x: estimate_rigid2d(x, fixed=ref_img), dtype='float32', drop_axis=(3), chunks=(1,4,4)).compute()
            len_dat = denoised_data.shape[0]
            trans_affine = rigid_interp(trans_affine, down_sample_registration, len_dat)
        # save trans_affs file
        np.save(f'{save_root}/trans_affs.npy', trans_affine)
    # load trans_affs file
    trans_affine_ = np.load(f'{save_root}/trans_affs.npy')
    trans_affine_ = da.from_array(trans_affine_, chunks=(1,4,4))
    print('--- Done registration reference image')

    # apply affine transform
    if not os.path.exists(f'{save_root}/motion_corrected_data.zarr'):
        # fix memory issue to load data all together for transpose on local machine
        # load data
        # swap axes
        splits_ = np.array_split(np.arange(num_t).astype('int'), num_t_chunks)
        print(f'Processing total {num_t_chunks} chunks in time.......')
        for nz, n_split in enumerate(splits_):
            if not os.path.exists(save_root+'/motion_corrected_data_chunks_%03d.zarr'%(nz)):
                print('Apply registration to rechunk layer %03d'%(nz))
                trans_data_ = da.map_blocks(apply_transform3d, denoised_data[n_split], trans_affine_[n_split], chunks=(1, *denoised_data.shape[1:]), dtype='float32')
                print('Starting to rechunk layer %03d'%(nz))
                trans_data_t_z = trans_data_.rechunk((-1, 1, chunks[1]//nsplit[0], chunks[2]//nsplit[1])).transpose((1, 2, 3, 0))
                trans_data_t_z.to_zarr(save_root+'/motion_corrected_data_chunks_%03d.zarr'%(nz))
                del trans_data_t_z
                gc.collect()
                print('finishing rechunking time chunk -- %03d of %03d'%(nz, num_t_chunks))

        print('Remove temporal files of registration')
        if os.path.exists(f'{save_root}/denoised_data.zarr'):
            shutil.rmtree(f'{save_root}/denoised_data.zarr')


        trans_data_t = da.concatenate([da.from_zarr(save_root+'/motion_corrected_data_chunks_%03d.zarr'%(nz)) for nz in range(num_t_chunks)], axis=-1)
        trans_data_t = trans_data_t.rechunk((1, chunks[1]//nsplit[0], chunks[2]//nsplit[1], -1))
        trans_data_t.to_zarr(f'{save_root}/motion_corrected_data.zarr')
        for nz in range(num_t_chunks):
            if os.path.exists(f'{save_root}/motion_corrected_data_chunks_%03d.zarr'%(nz)):
                print('Remove temporal files of registration at %03d'%(nz))
                shutil.rmtree(f'{save_root}/motion_corrected_data_chunks_%03d.zarr'%(nz))
    fdask.terminate_workers(cluster, client)
    return None


def detrend_data(dir_root, save_root, window=100, percentile=20, nsplit = (4, 4), dask_tmp=None, memory_limit=0):
    if not os.path.exists(f'{save_root}/detrend_data.zarr'):
        cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
        print_client_links(cluster)
        print('Compute detrend data ---')
        trans_data_t = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
        Y_d = trans_data_t.map_blocks(lambda v: v - baseline(v, window=window, percentile=percentile), dtype='float32')
        Y_d.to_zarr(f'{save_root}/detrend_data.zarr')
        del Y_d
        fdask.terminate_workers(cluster, client)
    return None


def default_mask(dir_root, save_root, dask_tmp=None, memory_limit=0):
    cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print_client_links(cluster)
    print('Compute default mask ---')
    Y = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
    Y_d = da.from_zarr(f'{save_root}/detrend_data.zarr')
#     Y_b = Y - Y_d
#     Y_b_min = Y_b.min(axis=-1, keepdims=True)
#     Y_b_min.to_zarr(f'{save_root}/Y_b_min.zarr', overwrite=True)
#     Y_b_max = Y_b.max(axis=-1, keepdims=True)
#     Y_b_max.to_zarr(f'{save_root}/Y_b_max.zarr', overwrite=True)
#     Y_b_max_mask = Y_b.max(axis=-1, keepdims=True)>2
#     Y_b_min_mask = Y_b.min(axis=-1, keepdims=True)>1
#     mask = Y_b_max_mask & Y_b_min_mask
#     mask.to_zarr(f'{save_root}/mask_map.zarr', overwrite=True)
    Y_d_max = Y_d.max(axis=-1, keepdims=True)
    Y_d_max.to_zarr(f'{save_root}/Y_d_max.zarr', overwrite=True)
    Y_max = Y.max(axis=-1, keepdims=True)
    Y_max.to_zarr(f'{save_root}/Y_max.zarr', overwrite=True)
    Y_ave = Y.mean(axis=-1, keepdims=True)
    Y_ave.to_zarr(f'{save_root}/Y_ave.zarr', overwrite=True)
#     Y_std = Y.std(axis=-1, keepdims=True)
#     Y_std.to_zarr(f'{save_root}/Y_std.zarr', overwrite=True)
    fdask.terminate_workers(cluster, client)
    return None


def local_pca_on_mask(save_root, is_dff=False, dask_tmp=None, memory_limit=0):
    cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print_client_links(cluster)
    Y_d = da.from_zarr(f'{save_root}/detrend_data.zarr')
    if is_dff:
        Y_t = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
        Y_d = Y_d/(Y_t - Y_d)
    mask = da.from_zarr(f'{save_root}/mask_map.zarr')
    Y_svd = da.map_blocks(fb_pca_block, Y_d, mask, dtype='float32')
    Y_svd.to_zarr(f'{save_root}/masked_local_pca_data.zarr', overwrite=True)
    fdask.terminate_workers(cluster, client)
    time.sleep(10)
    return None


def demix_cells(save_root, dt, is_skip=True, dask_tmp=None, memory_limit=0):
    '''
      1. local pca denoise
      2. cell segmentation
    '''
    cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print_client_links(cluster)
    Y_svd = da.from_zarr(f'{save_root}/detrend_data.zarr')
    Y_svd = Y_svd[:, :, :, ::dt]
    mask = da.from_zarr(f'{save_root}/Y_d_max.zarr')
    if not os.path.exists(f'{save_root}/sup_demix_rlt/'):
        os.mkdir(f'{save_root}/sup_demix_rlt/')
    da.map_blocks(demix_blocks, Y_svd, mask, chunks=(1, 1, 1, 1), dtype='int8', save_folder=save_root, is_skip=is_skip).compute()
    fdask.terminate_workers(cluster, client)
    time.sleep(10)
    return None


def check_fail_block(save_root, dt=0):
    file = glob(f'{save_root}/masked_local_pca_data.zarr/*.partial')
    print(file)


def check_demix_cells(save_root, block_id, plot_global=True, plot_mask=True, mask=None):
    import matplotlib.pyplot as plt
    Y_d_ave = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
    _, xdim, ydim, _ = Y_d_ave.shape
    _, x_, y_, _ = Y_d_ave.chunksize
    Y_d_ave_ = np.load(save_root+'Y_ave.npy')
    Y_d_ave_ = Y_d_ave_[block_id[0],block_id[1]*x_:block_id[1]*x_+x_, block_id[2]*y_:block_id[2]*y_+y_].squeeze()
    try:
        A_ = load_A_matrix(save_root=save_root, block_id=block_id, min_size=0)
        A_[A_<A_.max(axis=0, keepdims=True)*0.3]=0
        A_ = A_.reshape((x_, y_, -1), order="F")
        mask_ = mask[x_*block_id[1]:x_*(block_id[1]+1), y_*block_id[2]:y_*(block_id[2]+1)]
        A_[~mask_]=0
        A_ = A_[:, :, (A_>0).sum(axis=(0,1))>10]
        A_ = A_.reshape(x_*y_, -1, order="F")
        A_comp = np.zeros(A_.shape[0])
        A_comp[A_.sum(axis=-1)>0] = np.argmax(A_[A_.sum(axis=-1)>0, :], axis=-1) + 1
        A_comp[A_comp>0] = A_comp[A_comp>0]%20+1
        plt.figure(figsize=(8, 8))
        plt.imshow(Y_d_ave_, vmax=np.percentile(Y_d_ave_, 99), cmap='gray')
        plt.title('Components')
        plt.axis('off')
        plt.show()
        plt.figure(figsize=(8, 8))
        plt.imshow(A_comp.reshape(y_, x_).T, cmap=plt.cm.nipy_spectral_r)
        plt.title('Components')
        plt.axis('off')
        plt.show()
        plt.figure(figsize=(8, 8))
        plt.title('Weights')
        A_comp = A_.sum(axis=-1)
        plt.imshow(A_comp.reshape(y_, x_).T)
        plt.axis('off')
        plt.show()
    except:
        print('No components')
    if plot_global:
        Y_d_ave = np.load(save_root+'Y_ave.npy')
        area_mask = np.zeros((xdim, ydim)).astype('bool')
        area_mask[block_id[1]*x_:block_id[1]*x_+x_, block_id[2]*y_:block_id[2]*y_+y_]=True
        plt.figure(figsize=(16, 16))
        plt.imshow(Y_d_ave[block_id[0]].squeeze(), vmax=np.percentile(Y_d_ave[block_id[0]], 95))
        plt.imshow(area_mask, cmap='gray', alpha=0.3)
        plt.axis('off')
        plt.show()
    return None


def check_demix_cells_layer(save_root, nlayer, nsplit = (10, 16), mask=None):
    import matplotlib.pyplot as plt
    Y_d_ave = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
    _, xdim, ydim, _ = Y_d_ave.shape
    _, x_, y_, _ = Y_d_ave.chunksize
    A_mat = np.zeros((xdim, ydim))
    for nx in range(nsplit[0]):
        for ny in range(nsplit[1]):
            try:
                A_ = load_A_matrix(save_root=save_root, ext='', block_id=(nlayer, nx, ny, 0), min_size=0)
                A_[A_<A_.max(axis=0, keepdims=True)*0.5]=0
                A_ = A_.reshape((x_, y_, -1), order="F")
                mask_ = mask[x_*nx:x_*(nx+1), y_*ny:y_*(ny+1)]
                A_[~mask_]=0
                A_ = A_[:, :, (A_>0).sum(axis=(0,1))>10]
                A_mat[x_*nx:x_*(nx+1), y_*ny:y_*(ny+1)] = A_.sum(axis=-1)
            except:
                pass

    plt.figure(figsize=(8, 8))
    plt.imshow(A_mat)
    plt.title(f'Components {nlayer}')
    plt.axis('off')
    plt.show()
    return None


def compute_cell_dff_raw(save_root, mask, dask_tmp=None, memory_limit=0):
    '''
      1. local pca denoise (\delta F signal)
      2. baseline
      3. Cell weight matrix apply to denoise and baseline
      4. dff
    '''
    # set worker
    if not os.path.exists(f'{save_root}/cell_raw_dff'):
        os.mkdir(f'{save_root}/cell_raw_dff')
    cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print_client_links(cluster)
    trans_data_t = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
    if not os.path.exists(f'{save_root}/cell_raw_dff'):
        os.makedirs(f'{save_root}/cell_raw_dff')
    da.map_blocks(compute_cell_raw_dff, trans_data_t, mask, dtype='float32', chunks=(1, 1, 1, 1), save_root=save_root, ext='').compute()
    fdask.terminate_workers(cluster, client)
    return None


def combine_dff(save_root):
    '''
      1. local pca denoise (\delta F signal)
      2. baseline
      3. Cell weight matrix apply to denoise and baseline
      4. dff
    '''
    # set worker
    A_loc_list = []
    A_list = []
    dFF_list = []
    A_shape = []
    for _ in glob(save_root+'cell_raw_dff/period_Y_demix_block_*.h5'):
        try:
            _ = File(_)
        except:
            continue
        A_loc = _['A_loc'].value
        A = _['A'].value
        dFF = _['cell_F'].value
        for n_ in range(A.shape[-1]):
            if np.abs(dFF[n_]).sum()>0:
                A_tmp = np.zeros((100, 100))
                A_loc_list.append(A_loc)
                x_, y_ = A[:, :, n_].shape
                A_shape.append(np.array([x_, y_]))
                A_tmp[:x_, :y_] = A[:, :, n_]
                A_list.append(A_tmp)
                dFF_list.append(dFF[n_])
    np.savez(save_root+'cell_raw_dff', A_loc=np.array(A_loc_list), A_shape=np.array(A_shape), \
             A=np.array(A_list), F=np.array(dFF_list))
    return None


def combine_dff_sparse(save_root):
    '''
      1. local pca denoise (\delta F signal)
      2. baseline
      3. Cell weight matrix apply to denoise and baseline
      4. dff
    '''
    # set worker
    A_loc_list = []
    A_list = []
    dFF_list = []
    A_shape = []
    for _ in glob(save_root+'cell_raw_dff/period_Y_demix_block_*.h5'):
        try:
            _ = File(_)
        except:
            continue
        A_loc = _['A_loc'].value
        try:
            A = _['A_s'].value
            dFF = _['cell_F_s'].value
            for n_ in range(A.shape[-1]):
                if np.abs(dFF[n_]).sum()>0:
                    A_tmp = np.zeros((100, 100))
                    A_loc_list.append(A_loc)
                    x_, y_ = A[:, :, n_].shape
                    A_shape.append(np.array([x_, y_]))
                    A_tmp[:x_, :y_] = A[:, :, n_]
                    A_list.append(A_tmp)
                    dFF_list.append(dFF[n_])
        except:
            pass
    np.savez(save_root+'cell_raw_dff_sparse', A_loc=np.array(A_loc_list), A_shape=np.array(A_shape), \
             A=np.array(A_list), F=np.array(dFF_list))
    return None