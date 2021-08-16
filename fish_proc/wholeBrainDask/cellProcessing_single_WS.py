import numpy as np
import os, sys, gc, shutil, time
from glob import glob
from h5py import File
import warnings
warnings.filterwarnings('ignore')
import dask
# dask.config.set({"jobqueue.lsf.use-stdin": True})
import dask.array as da
from .utils import *
from ..utils import dask_ as fdask
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'


def print_client_links(cluster):
    print(f'Scheduler: {cluster.scheduler_address}')
    print(f'Dashboard link: {cluster.dashboard_link}')
    return None


def preprocessing(dir_root, save_root, cameraNoiseMat=cameraNoiseMat, nsplit = (4, 4), num_t_chunks = 80,\
                  dask_tmp=None, memory_limit=0, is_singlePlane=False, down_sample_registration=1):
    from ..utils.getCameraInfo import getCameraInfo
    from tqdm import tqdm
    from ..utils.fileio import du
    
    if isinstance(save_root, list):
        save_root_ext = save_root[1]
        save_root = save_root[0]
    
    print(f'Tmp files will be saved to {save_root}')
    if 'save_root_ext' in locals():
        print(f'With extended drive to {save_root_ext}')
    print(f'is_singlePlane: {is_singlePlane}')
    print(f'nsplit: {nsplit}')

    if not os.path.exists(f'{save_root}/denoised_data.zarr'):
        # set worker
        cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
        print_client_links(cluster)
        print('========================')
        print('Getting data infos')
        files = sorted(glob(dir_root+'/*.h5'))
        chunks = File(files[0],'r')['default'].shape
        cameraInfo = getCameraInfo(dir_root)
        print('Stacking data')
        imread = dask.delayed(lambda v: pixelDenoiseImag(File(v,'r')['default'].value, cameraNoiseMat=cameraNoiseMat, cameraInfo=cameraInfo))
        
        num_files = len(files)
        splits_ = np.array_split(np.arange(num_files).astype('int'), num_t_chunks)
        npfiles=np.array(files)
        
        for nz, n_split in enumerate(splits_):
            if not os.path.exists(save_root+'/denoised_data_%03d.zarr'%(nz)):
                print('Apply denoising to file chunk %03d'%(nz))
                lazy_data = [imread(fn) for fn in npfiles[n_split]]
                sample = lazy_data[0].compute()
                if not is_singlePlane:
                    denoised_data = da.stack([da.from_delayed(fn, shape=sample.shape, dtype=sample.dtype) for fn in lazy_data])
                else:
                    if len(chunks)==2:
                        denoised_data = da.stack([da.from_delayed(fn, shape=sample.shape, dtype=sample.dtype) for fn in lazy_data])
                    else:
                        denoised_data = da.concatenate([da.from_delayed(fn, shape=sample.shape, dtype=sample.dtype) for fn in lazy_data], axis=0).rechunk((1,-1,-1))
                denoised_data.to_zarr(save_root+'/denoised_data_%03d.zarr'%(nz))
                print('finishing denoising chunk -- %03d of %03d'%(nz, num_t_chunks))
                f = open(f'{save_root}/processing.tmp', "a")
                f.write('finishing denoised data chunk -- %03d of %03d \n'%(nz, num_t_chunks))
                f.close()
        denoised_data = da.concatenate([da.from_zarr(save_root+'/denoised_data_%03d.zarr'%(nz)) for nz in range(num_t_chunks)])
        denoised_data.to_zarr(f'{save_root}/denoised_data.zarr')    
        def rm_tmp(nz, save_root=save_root):
            if os.path.exists(f'{save_root}/denoised_data_%03d.zarr'%(nz)):
                print('Remove temporal files of denoise at %03d'%(nz))
                shutil.rmtree(f'{save_root}/denoised_data_%03d.zarr'%(nz))
            return np.array([1])    
        nz_list = da.from_array(np.arange(num_t_chunks), chunks=(1)) 
        da.map_blocks(rm_tmp, nz_list).compute()
        
#         lazy_data = [imread(fn) for fn in files]
#         sample = lazy_data[0].compute()
#         if not is_singlePlane:
#             denoised_data = da.stack([da.from_delayed(fn, shape=sample.shape, dtype=sample.dtype) for fn in lazy_data])
#         else:
#             if len(chunks)==2:
#                 denoised_data = da.stack([da.from_delayed(fn, shape=sample.shape, dtype=sample.dtype) for fn in lazy_data])
#             else:
#                 denoised_data = da.concatenate([da.from_delayed(fn, shape=sample.shape, dtype=sample.dtype) for fn in lazy_data], axis=0).rechunk((1,-1,-1))        
#         print('========================')
#         print('Denoising camera noise')
#         print('Denoising camera noise -- save data')
#         denoised_data.to_zarr(f'{save_root}/denoised_data.zarr')
        fdask.terminate_workers(cluster, client)
        time.sleep(30)
    
    # set worker
    cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print_client_links(cluster)
    print('Denoising camera noise -- load saved data')
    f = open(f'{save_root}/processing.tmp', "a")
    f.write(f'Denoising camera noise -- load saved data \n')
    f.close()
    denoised_data = da.from_zarr(f'{save_root}/denoised_data.zarr')
    if denoised_data.ndim==3:
        denoised_data = denoised_data[:, None, :, :]
    chunks = denoised_data.shape[1:]
    num_t = denoised_data.shape[0]

    # save and compute reference image
    print('Compute reference image ---')
    f = open(f'{save_root}/processing.tmp', "a")
    f.write(f'Compute reference image --- \n')
    f.close()
    if not os.path.exists(f'{save_root}/motion_fix_.h5'):
        med_win = len(denoised_data)//2
        ref_img = denoised_data[med_win-50:med_win+50].mean(axis=0).compute()
        save_h5(f'{save_root}/motion_fix_.h5', ref_img, dtype='float16')

    print('--- Done computing reference image')
    f = open(f'{save_root}/processing.tmp', "a")
    f.write(f'--- Done computing reference image \n')
    f.close()

    # compute affine transform
    print('Registration to reference image ---')
    f = open(f'{save_root}/processing.tmp', "a")
    f.write(f'Registration to reference image --- \n')
    f.close()
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
    trans_affine_ = da.from_array(np.expand_dims(trans_affine_, 3), chunks=(1,4,4,1))
    print('--- Done registration reference image')
    f = open(f'{save_root}/processing.tmp', "a")
    f.write(f'--- Done registration reference image \n')
    f.close()
    fdask.terminate_workers(cluster, client)
    time.sleep(30)

    # apply affine transform
    if not os.path.exists(f'{save_root}/motion_corrected_data.zarr'):
        # fix memory issue to load data all together for transpose on local machine
        # load data
        # swap axes

        # set worker
        cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
        print_client_links(cluster)

        splits_ = np.array_split(np.arange(num_t).astype('int'), num_t_chunks)
        print(f'Processing total {num_t_chunks} chunks in time.......')
        # estimate size of data to store
        used_ = du(f'{save_root}/denoised_data.zarr/')
        est_data_size = int(used_.decode('utf-8'))//(2**20*num_t_chunks*2)+5 #kb to Gb
        for nz, n_split in enumerate(splits_):
            if not os.path.exists(save_root+'/motion_corrected_data_chunks_%03d.zarr'%(nz)):
                if 'save_root_ext' in locals():
                    if os.path.exists(save_root_ext+'/motion_corrected_data_chunks_%03d.zarr'%(nz)):
                        continue
                print('Apply registration to rechunk layer %03d'%(nz))
                t_start = n_split[0]
                t_end = n_split[-1]+1
                trans_data_ = da.map_blocks(apply_transform3d, denoised_data[t_start:t_end], trans_affine_[t_start:t_end], chunks=(1, *denoised_data.shape[1:]), dtype='float16')
                print('Starting to rechunk layer %03d'%(nz))
                trans_data_t_z = trans_data_.rechunk((-1, 1, chunks[1]//nsplit[0], chunks[2]//nsplit[1])).transpose((1, 2, 3, 0))
                # check space availablity
                _, _, free_ = shutil.disk_usage(f'{save_root}/')
                if (free_//(2**30)) > est_data_size:
                    print(f'Remaining space {free_//(2**30)} GB..... -- start to save at {save_root}')
                    trans_data_t_z.to_zarr(save_root+'/motion_corrected_data_chunks_%03d.zarr'%(nz))
                else:
                    try:
                        print(f'Remaining space {free_//(2**30)} GB..... -- start to save at {save_root_ext}')
                        trans_data_t_z.to_zarr(save_root_ext+'/motion_corrected_data_chunks_%03d.zarr'%(nz))
                    except Exception as e:
                        # if any error -- break the code
                        print(e)    
                        fdask.terminate_workers(cluster, client)
                        return None
                del trans_data_t_z
                gc.collect()
                print('finishing rechunking time chunk -- %03d of %03d'%(nz, num_t_chunks))
                f = open(f'{save_root}/processing.tmp', "a")
                f.write('finishing rechunking time chunk -- %03d of %03d \n'%(nz, num_t_chunks))
                f.close()

        print('Remove temporal files of registration')
        if os.path.exists(f'{save_root}/denoised_data.zarr'):
            shutil.rmtree(f'{save_root}/denoised_data.zarr')
        for ext_files in tqdm(glob(save_root_ext+'/motion_corrected_data_chunks_*.zarr')):
            print(f'Moving file {ext_files} to Tmp-file folder.....')
            shutil.move(ext_files, save_root+'/')
        fdask.terminate_workers(cluster, client)
        time.sleep(60)
    return None


def combine_preprocessing(dir_root, save_root, num_t_chunks = 80, dask_tmp=None, memory_limit=0):
    cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print_client_links(cluster)
    # chunks = da.from_zarr(save_root+'/motion_corrected_data_chunks_%03d.zarr'%(0)).chunksize
    trans_data_t = da.concatenate([da.from_zarr(save_root+'/motion_corrected_data_chunks_%03d.zarr'%(nz)) for nz in range(num_t_chunks)], axis=-1)
    # trans_data_t = trans_data_t.rechunk((1, chunks[1], chunks[2], -1))
    trans_data_t.to_zarr(f'{save_root}/motion_corrected_data.zarr')    
    def rm_tmp(nz, save_root=save_root):
        if os.path.exists(f'{save_root}/motion_corrected_data_chunks_%03d.zarr'%(nz)):
            print('Remove temporal files of registration at %03d'%(nz))
            shutil.rmtree(f'{save_root}/motion_corrected_data_chunks_%03d.zarr'%(nz))
        return np.array([1])    
    nz_list = da.from_array(np.arange(num_t_chunks), chunks=(1)) 
    da.map_blocks(rm_tmp, nz_list).compute()
    fdask.terminate_workers(cluster, client)
    time.sleep(60)
    return None


def preprocessing_cluster(dir_root, save_root, cameraNoiseMat=cameraNoiseMat, nsplit = (4, 4), num_t_chunks = 80,\
                  dask_tmp=None, memory_limit=0, is_bz2=False, is_singlePlane=False, down_sample_registration=1):
    from ..utils.getCameraInfo import getCameraInfo
    # set worker
    cluster, client = fdask.setup_workers(numCore=200, is_local=False, dask_tmp=dask_tmp, memory_limit=memory_limit)
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

    trans_data_ = da.map_blocks(apply_transform3d, denoised_data, trans_affine_, chunks=(1, *denoised_data.shape[1:]), dtype='float16')
    trans_data_t = trans_data_.rechunk((-1, 1, chunks[1]//nsplit[0], chunks[2]//nsplit[1])).transpose((1, 2, 3, 0))
    trans_data_t.to_zarr(f'{save_root}/motion_corrected_data.zarr')
    fdask.terminate_workers(cluster, client)

    print('Remove temporal files of registration')
    if os.path.exists(f'{save_root}/denoised_data.zarr'):
        shutil.rmtree(f'{save_root}/denoised_data.zarr')
    return None


def detrend_data(dir_root, save_root, window=100, percentile=20, dask_tmp=None, memory_limit=0):
    if not os.path.exists(f'{save_root}/detrend_data.zarr'):
        cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
        print_client_links(cluster)
        print('Compute detrend data ---')
        trans_data_t = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
        Y_d = trans_data_t.map_blocks(lambda v: v - baseline(v, window=window, percentile=percentile, downsample=window//40), dtype='float16')
        Y_d.to_zarr(f'{save_root}/detrend_data.zarr')
        del Y_d
        fdask.terminate_workers(cluster, client)
        time.sleep(60)
    return None


def default_mask(dir_root, save_root, dask_tmp=None, memory_limit=0):
    cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print_client_links(cluster)
    print('Compute default mask ---')
    Y = da.from_zarr(f'{save_root}/motion_corrected_data.zarr')
    Y_d = da.from_zarr(f'{save_root}/detrend_data.zarr')
    Y_d_max = Y_d.max(axis=-1, keepdims=True)
    Y_d_max.to_zarr(f'{save_root}/Y_d_max.zarr', overwrite=True)
    Y_max = Y.max(axis=-1, keepdims=True)
    Y_max.to_zarr(f'{save_root}/Y_max.zarr', overwrite=True)
    Y_ave = Y.astype('float').mean(axis=-1, keepdims=True).astype(Y.dtype)
    Y_ave.to_zarr(f'{save_root}/Y_ave.zarr', overwrite=True)
    fdask.terminate_workers(cluster, client)
    time.sleep(100)
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
    time.sleep(60)
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
    da.map_blocks(demix_blocks, Y_svd.astype('float'), mask.astype('float'), chunks=(1, 1, 1, 1), dtype='int8', save_folder=save_root, is_skip=is_skip).compute()
    fdask.terminate_workers(cluster, client)
    time.sleep(100)
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
    da.map_blocks(compute_cell_raw_dff, trans_data_t.astype('float'), mask, dtype='float32', chunks=(1, 1, 1, 1), save_root=save_root, ext='').compute()
    fdask.terminate_workers(cluster, client)
    time.sleep(100)
    return None


def combine_dff(save_root):
    '''
      1. local pca denoise (\delta F signal)
      2. baseline
      3. Cell weight matrix apply to denoise and baseline
      4. dff
    '''
    from tqdm import tqdm
    # set worker
    A_loc_list = []
    A_list = []
    dFF_list = []
    A_shape = []
    for _ in tqdm(glob(save_root+'cell_raw_dff/period_Y_demix_block_*.h5')):
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
    np.savez(save_root+'cell_raw_dff', \
             A_loc=np.array(A_loc_list).astype('int16'), \
             A_shape=np.array(A_shape).astype('int16'), \
             A=np.array(A_list).astype('float16'), \
             F=np.array(dFF_list).astype('float16'))
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
    from tqdm import tqdm
    for _ in tqdm(glob(save_root+'cell_raw_dff/period_Y_demix_block_*.h5')):
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
    np.savez(save_root+'cell_raw_dff_sparse', \
             A_loc=np.array(A_loc_list).astype('int16'), \
             A_shape=np.array(A_shape).astype('int16'), \
             A=np.array(A_list).astype('float16'), \
             F=np.array(dFF_list).astype('float16'))
    return None


def compute_dff_from_f(save_root, min_F=20, window=400, percentile=20, downsample=10, dFF_max=5, dask_tmp=None, memory_limit=0):
    cluster, client = fdask.setup_workers(is_local=True, dask_tmp=dask_tmp, memory_limit=memory_limit)
    print_client_links(cluster)
    
    _ = np.load(save_root+'cell_raw_dff_sparse.npz', allow_pickle=True)
    A = _['A'].astype('float')
    F_ = _['F'].astype('float')
    A_loc = _['A_loc']
    _ = None
    valid_cell = F_.max(axis=-1)>min_F
    A = A[valid_cell]
    A_loc = A_loc[valid_cell]
    F_ = F_[valid_cell]
    F_dask = da.from_array(F_, chunks=('auto', -1))
    baseline_ = da.map_blocks(fwc.baseline, F_dask, dtype='float', window=window, percentile=percentile, downsample=downsample).compute()
    dFF = F_/baseline_-1
    invalid_ = (dFF.max(axis=-1)>dFF_max) | (np.isnan(dFF.max(axis=-1))) | (baseline_.min(axis=-1)<=0)
    np.savez(save_root+'cell_dff.npz', A=A[~invalid_].astype('float16'), A_loc=A_loc[~invalid_], dFF=dFF[~invalid_].astype('float16'))
    
    fdask.terminate_workers(cluster, client)
    time.sleep(100)
    return None