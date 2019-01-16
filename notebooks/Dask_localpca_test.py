"""
Using dask

@author: modified by Salma Elmalaki, Jan 2019

"""

import numpy as np
import os, sys
from glob import glob
from skimage import io
import dask.array as da
import dask.array.image as dai
import dask_ndfilters as daf
import dask.dataframe as dd
import dask.bag as db
import dask

import time
from os.path import exists

dat_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/ProcessedData/'
cameraNoiseMat = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/gainMat/gainMat20180208'
root_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/10182018/Fish3-2/'
save_folder = dat_folder + '10182018/Fish3-2/DataSample/using_100_samples/'
#save_folder = dat_folder + '10182018/Fish3-2/Data/backup_before_improvements/'    #All samples #regular
#save_folder = dat_folder + '10182018/Fish3-2/Data/'  # Allsamples improved
sample_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/10182018/Fish3-2/test_sample/'  # Currently 20 samples



#################################### Utils ########################################
def load_img_seq_dask(img_folder):
    imread = dask.delayed(io.imread, pure=True)  # Lazy version of imread
    imgFiles = sorted(glob(img_folder + 'Y_dt*.tif'))
    #start_time = time.time()
    lazy_images = [imread(img).astype('float32') for img in imgFiles]  # Lazily evaluate imread on each path
    sample = lazy_images[0].compute()  # load the first image (assume rest are same shape/dtype)
    arrays = [da.from_delayed(lazy_image,  # Construct a small Dask array
                              dtype=sample.dtype,  # for every lazy value
                              shape=sample.shape)
              for lazy_image in lazy_images]

    imgStack = da.concatenate(arrays, axis=-1).astype('float32')
    imgStack = imgStack.compute()
    #print("--- %s seconds for image stack creation: dask ---" % (time.time() - start_time))
    return imgStack



def denose_2dsvd(Y_d, fishName, nblocks=[10, 10], stim_knots=None, stim_delta=0, ext=''):
    from fish_proc.denoiseLocalPCA.denoise import temporal as svd_patch
    from fish_proc.utils.memory import get_process_memory, clear_variables
    from skimage.external.tifffile import imsave, imread

    dx=4
    maxlag=5
    confidence=0.99
    greedy=False,
    fudge_factor=1
    mean_th_factor=1.15
    U_update=False
    min_rank=1

    Y_svd, _ = svd_patch(Y_d, nblocks=nblocks, dx=dx, stim_knots=stim_knots, stim_delta=stim_delta)
    np.save(f'{fishName}/Y_2dsvd{ext}', Y_svd.astype('float32'))

    #print(" ---- Y_svd shape ---"+str(Y_svd.shape) )
    #print(Y_svd.shape)
    # Salme: save the tif at the same time for later processing
    start_time = time.time()
    imsave(f'{fishName}/Y_2dsvd{ext}.tif', Y_svd.astype('float32'), compress=1)
    print("--- %s seconds for save Y_2dsvd.tif---" % (time.time() - start_time))

    #Y_svd = None
    #clear_variables(Y_svd)
    #get_process_memory()
    #return None
    return Y_svd

#################################### Regular ########################################
def local_pca():
    from pathlib import Path
    from skimage.external.tifffile import imsave, imread

    if not os.path.isfile(save_folder+'/proc_local_denoise.tmp'):
        if os.path.isfile(save_folder+'/finished_detrend.tmp'):
            Path(save_folder+'/proc_local_denoise.tmp').touch()

            start_time = time.time()
            if os.path.isfile(f'{save_folder}/Y_d.npy'):
                Y_d = np.load(f'{save_folder}/Y_d.npy').astype('float32')
            elif os.path.isfile(f'{save_folder}/Y_d.tif'):
                Y_d = imread(f'{save_folder}/Y_d.tif')
            print("--- %s seconds for read---" % (time.time() - start_time))

            start_time = time.time()
            Y_d_ave = Y_d.mean(axis=-1, keepdims=True) # remove mean
            Y_d_std = Y_d.std(axis=-1, keepdims=True) # normalization
            Y_d = (Y_d - Y_d_ave)/Y_d_std
            Y_d = Y_d.astype('float32')
            print("--- %s seconds for normalization---" % (time.time() - start_time))

            start_time = time.time()
            np.savez_compressed(f'{save_folder}/Y_2dnorm', Y_d_ave=Y_d_ave, Y_d_std=Y_d_std)
            print("--- %s seconds for save y norm---" % (time.time() - start_time))

            Y_d_ave = None
            Y_d_std = None

            start_time = time.time()
            for n, Y_d_ in enumerate(np.array_split(Y_d, 10, axis=-1)):
                denose_2dsvd(Y_d_, save_folder, ext=f'{n}')
                print(str(n)+"--- %s seconds for denose_2dsvd---"  % (time.time() - start_time))

            print("--- %s seconds for total denose_2dsvd---" % (time.time() - start_time))

            Y_d_ = None
            Y_d = None

        Path(save_folder+'/finished_local_denoise.tmp').touch()
    return None


#################################### Multiple load and save ########################################
def local_pca_multiple():
    from pathlib import Path
    from skimage.external.tifffile import imsave, imread


    save_folder_Detrend = save_folder + 'Detrend/'
    save_folder_LocalPCA = save_folder + 'LocalPCA/divide10block10x10/svdsparse/'


    if not os.path.isfile(save_folder_LocalPCA + '/proc_local_denoise.tmp'):
        if os.path.isfile(save_folder_Detrend + '/finished_detrend.tmp'):
            Path(save_folder_LocalPCA + '/proc_local_denoise.tmp').touch()

            start_time = time.time()
            if os.path.isfile(f'{save_folder_Detrend}/Y_d.npy'):
                Y_d = np.load(f'{save_folder_Detrend}/Y_d.npy').astype('float32')
            #elif os.path.isfile(f'{save_folder_Detrend}/Y_d.tif'):
            else:
                Y_d = load_img_seq_dask(save_folder_Detrend)
                np.save(f'{save_folder_Detrend}/Y_d.npy', Y_d.astype('float32') ) #save it for future

            print("--- %s seconds for read---" % (time.time() - start_time))

            start_time = time.time()
            Y_d_ave = Y_d.mean(axis=-1, keepdims=True)  # remove mean
            Y_d_std = Y_d.std(axis=-1, keepdims=True)   # normalization
            Y_d = (Y_d - Y_d_ave) / Y_d_std
            Y_d = Y_d.astype('float32')
            print("--- %s seconds for normalization---" % (time.time() - start_time))

            start_time = time.time()
            np.savez_compressed(f'{save_folder_LocalPCA}/Y_2dnorm', Y_d_ave=Y_d_ave, Y_d_std=Y_d_std)
            print("--- %s seconds for save y norm---" % (time.time() - start_time))

            Y_d_ave = None
            Y_d_std = None

            start_time = time.time()
            n_splits = 10 #10  #salma
            nblocks = [10, 10] #[10, 10]
            #Y_svd = [] # added by salma
            for n, Y_d_ in enumerate(np.array_split(Y_d, n_splits, axis=-1)):
                Y_2dsvd = denose_2dsvd(Y_d_, save_folder_LocalPCA, nblocks, ext=f'{n}')
                #Y_svd.append(Y_2dsvd) #added by salma
                Y_2dsvd = None
                print(str(n) + "--- %s seconds for denose_2dsvd---" % (time.time() - start_time))



            # batches = []
            # for index, img_split in enumerate(np.array_split(Y_d, 10, axis=-1)):
            #     result_batch = dask.delayed(denose_2dsvd)(img_split, save_folder_LocalPCA, ext=f'{index}')
            #     batches.append(result_batch)
            #
            # dask.compute(*batches)


            print("--- %s seconds for denose_2dsvd---" % (time.time() - start_time))

            # salma added save here
            # start_time = time.time()
            # Y_svd = np.concatenate(Y_svd, axis=-1)
            # print(Y_svd.shape)
            # #imsave(f'{save_folder_LocalPCA}/Y_svd.tif', Y_svd.astype('float32'), compress=1) #this takes too much time
            # np.save(f'{save_folder_LocalPCA}/Y_svd', Y_svd.astype('float32'))
            # print('Concatenate files into a tif and np file')
            # print("--- %s seconds for save Y_svd---" % (time.time() - start_time))


            Y_svd = None
            Y_d_ = None
            Y_d = None

            Path(save_folder_LocalPCA + '/finished_local_denoise.tmp').touch()
    return None

######### Tests and main #########

## python Dask_registration_test.py
if __name__ == '__main__':

    import random
    #salma fix the compare the resuls
    np.random.seed(0)

    if len(sys.argv)>1:
        ext = ''
        if len(sys.argv)>2:
            ext = sys.argv[2]
        eval(sys.argv[1]+f"({ext})")
    else:
        start_time = time.time()
        local_pca_multiple()
        #local_pca()
        print("--- %s seconds for local pca---" % (time.time() - start_time))
