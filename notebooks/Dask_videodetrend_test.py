"""
Using dask

@author: modified by Salma Elmalaki, Dec 2018

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
#save_folder = dat_folder + '10182018/Fish3-2/Data/backup_before_improvements/'   #All samples
sample_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/10182018/Fish3-2/test_sample/'  # Currently 20 samples


#################################### Utils ########################################
def load_img_seq_dask(img_folder):
    imread = dask.delayed(io.imread, pure=True)  # Lazy version of imread
    imgFiles = sorted(glob(img_folder + 'imgDMotion*.tif'))
    #start_time = time.time()
    lazy_images = [imread(img).astype('float32') for img in imgFiles]  # Lazily evaluate imread on each path
    sample = lazy_images[0].compute()  # load the first image (assume rest are same shape/dtype)
    arrays = [da.from_delayed(lazy_image,  # Construct a small Dask array
                              dtype=sample.dtype,  # for every lazy value
                              shape=sample.shape)
              for lazy_image in lazy_images]

    imgStack = da.concatenate(arrays, axis=0).astype('float32')
    imgStack = imgStack.compute()
    #print("--- %s seconds for image stack creation: dask ---" % (time.time() - start_time))
    return imgStack




#################################### Regular #######################################
def detrend(Y_, fishName, n_split = 1, ext=''):
    from fish_proc.denoiseLocalPCA.detrend import trend
    from fish_proc.utils.np_mp import parallel_to_chunks
    from fish_proc.utils.memory import get_process_memory, clear_variables

    Y_trend = []
    for Y_split in np.array_split(Y_, n_split, axis=0):
        Y_trend.append(parallel_to_chunks(trend, Y_split.astype('float32'))[0].astype('float32'))

    # Y_trend = parallel_to_chunks(trend, Y_split)
    # Y_trend = Y_trend[0]
    # Y_trend_ = tuple([_ for _ in Y_trend])
    # Y_trend_ = np.concatenate(Y_trend_, axis=0)
    Y_trend = np.concatenate(Y_trend, axis=0).astype('float32')
    # Y_d = Y_ - Y_trend
    np.save(f'{fishName}/Y_d{ext}', Y_ - Y_trend)
    # np.save(f'{fishName}/Y_trend{ext}', Y_trend)
    # Y_d = None
    Y_split = None
    Y_trend = None
    clear_variables((Y_split, Y_, Y_trend))
    return None


def video_detrend():
    #from fish_proc.pipeline.denoise_dask import detrend
    from pathlib import Path
    from multiprocessing import cpu_count
    from skimage.io import imsave, imread



    if os.path.isfile(save_folder+'/finished_detrend.tmp'):
        return

    if not os.path.isfile(save_folder+'/Y_d.tif') and not os.path.isfile(save_folder+'/proc_detrend.tmp'):
        if os.path.isfile(save_folder+'/finished_registr.tmp'):
            Path(save_folder+'/proc_detrend.tmp').touch()

            start_time = time.time()
            Y = imread(save_folder+'/imgDMotion.tif').astype('float32')
            print(Y.shape[0])
            print("--- %s seconds for read---" % (time.time() - start_time))

            #Y = Y.reshape(Y.shape[0], Y.shape[2], Y.shape[3]) # Just in the testing part

            Y = Y.transpose([1,2,0])
            print("--- %s seconds for transpose---" % (time.time() - start_time))
            n_split = min(Y.shape[0]//cpu_count(), 8)
            if n_split <= 1:
                n_split = 2
            Y_len = Y.shape[0]//2
            detrend(Y[:Y_len, :, :], save_folder, n_split=n_split//2, ext='0')
            print("--- %s seconds for detrend1---" % (time.time() - start_time))
            detrend(Y[Y_len:, :, :], save_folder, n_split=n_split//2, ext='1')
            print("--- %s seconds for detrend2---" % (time.time() - start_time))
            Y = None
            Y = []
            Y.append(np.load(save_folder+'/Y_d0.npy').astype('float32'))
            Y.append(np.load(save_folder+'/Y_d1.npy').astype('float32'))
            print("--- %s seconds for loading 1 and 2---" % (time.time() - start_time))
            Y = np.concatenate(Y, axis=0).astype('float32')
            imsave(save_folder+'/Y_d.tif', Y, compress=1)
            print("--- %s seconds for save---" % (time.time() - start_time))
            Y = None
            os.remove(save_folder+'/Y_d0.npy')
            os.remove(save_folder+'/Y_d1.npy')
            # os.remove(save_folder+'/Y_trend0.npy')
            # os.remove(save_folder+'/Y_trend1.npy')
            Path(save_folder+'/finished_detrend.tmp').touch()
    return None



#################################### Regular Multiple save and load #######################################
def video_detrend_multiple():
    #from fish_proc.pipeline.denoise_dask import detrend
    from pathlib import Path
    from multiprocessing import cpu_count
    from skimage.io import imsave, imread

    save_folder_Registration = save_folder + 'Registration/'
    save_folder_Detrend = save_folder + 'Detrend/'

    if os.path.isfile(save_folder_Detrend+'finished_detrend.tmp'):
        return

    if not os.path.isfile(save_folder_Detrend+'Y_d.tif') and not os.path.isfile(save_folder_Detrend+'proc_detrend.tmp'):
        if os.path.isfile(save_folder_Registration+'finished_registr.tmp'):
            Path(save_folder_Detrend+'proc_detrend.tmp').touch()
            start_time = time.time()

            # multiple load
            Y = load_img_seq_dask(save_folder_Registration)
            print("--- %s seconds for read---" % (time.time() - start_time))

            print(Y.shape)
            #Y = Y.reshape(Y.shape[0], Y.shape[2], Y.shape[3]) # Just in the testing part
            Y = Y.transpose([1,2,0])
            print("--- %s seconds for transpose---" % (time.time() - start_time))


            n_split = min(Y.shape[0]//cpu_count(), 8)
            if n_split <= 1:
                n_split = 2
            Y_len = Y.shape[0]//2
            detrend(Y[:Y_len, :, :], save_folder_Detrend, n_split=n_split//2, ext='0')
            print("--- %s seconds for detrend1---" % (time.time() - start_time))
            detrend(Y[Y_len:, :, :], save_folder_Detrend, n_split=n_split//2, ext='1')
            print("--- %s seconds for detrend2---" % (time.time() - start_time))
            Y = None
            Y = []
            Y.append(np.load(save_folder_Detrend+'/Y_d0.npy').astype('float32'))
            Y.append(np.load(save_folder_Detrend+'/Y_d1.npy').astype('float32'))
            print("--- %s seconds for loading 1 and 2---" % (time.time() - start_time))
            Y = np.concatenate(Y, axis=0).astype('float32')
            print("--- %s seconds for concat---" % (time.time() - start_time))

            # multiple save
            n_splits = Y.shape[2] // 50
            imgSplit = np.split(Y, n_splits, axis = 2)
            delayed_imsave = dask.delayed(io.imsave, pure=True)  # Lazy version of imsave
            lazy_images = [
                delayed_imsave(save_folder_Detrend + '/Y_dt' + '%04d'%index   + '.tif', img, compress=1) for
                index, img in enumerate(imgSplit)]  # Lazily evaluate imsave on each path
            dask.compute(*lazy_images)
            print("--- %s seconds for save dask ---" % (time.time() - start_time))  # --salma

            Y = None
            imgSplit = None
            os.remove(save_folder_Detrend+'/Y_d0.npy')
            os.remove(save_folder_Detrend+'/Y_d1.npy')
            # os.remove(save_folder+'/Y_trend0.npy')
            # os.remove(save_folder+'/Y_trend1.npy')
            Path(save_folder_Detrend+'/finished_detrend.tmp').touch()
    return None



######### Tests and main #########

## python Dask_registration_test.py
if __name__ == '__main__':
    if len(sys.argv)>1:
        ext = ''
        if len(sys.argv)>2:
            ext = sys.argv[2]
        eval(sys.argv[1]+f"({ext})")
    else:
        start_time = time.time()
        #video_detrend()
        video_detrend_multiple()
        print("--- %s seconds for video detrend---" % (time.time() - start_time))
