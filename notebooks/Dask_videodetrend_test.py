"""
Using dask

@author: modified by Salma Elmalaki, Dec 2018

"""


import numpy as np
import pandas as pd
import os, sys
from glob import glob
from skimage import io
import dask.array as da
import dask.array.image as dai
import dask_ndfilters as daf
import dask.dataframe as dd
import dask.bag as db
import dask
from numba import vectorize

import time
from os.path import exists
import random


dat_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/ProcessedData/'
cameraNoiseMat = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/gainMat/gainMat20180208'
root_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/10182018/Fish3-2/'
save_folder = dat_folder + '10182018/Fish3-2/DataSample/using_20_samples/'
sample_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/10182018/Fish3-2/test_sample/'  # Currently 20 samples


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

            Y = imread(save_folder+'/imgDMotion.tif').astype('float32')
            print(Y.shape[0])
            Y = Y.reshape(Y.shape[0], Y.shape[2], Y.shape[3]) # Just in the testing part
            start_time = time.time()
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


######### Tests and main #########

## python Dask_registration_test.py
if __name__ == '__main__':
    if len(sys.argv)>1:
        ext = ''
        if len(sys.argv)>2:
            ext = sys.argv[2]
        eval(sys.argv[1]+f"({ext})")
    else:
        video_detrend()

