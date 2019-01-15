"""
Using dask

@author: modified by Salma Elmalaki, Jan 2019

"""

import numpy as np
import pandas as pd
from enum import Enum
import os, sys
from fish_proc.utils.memory import get_process_memory, clear_variables
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
#save_folder = dat_folder + '10182018/Fish3-2/DataSample/using_100_samples/'
save_folder = dat_folder + '10182018/Fish3-2/Data/'    #All samples
#save_folder = dat_folder + '10182018/Fish3-2/Data/backup_before_improvements/' # This is the regular run with 10block10x10
sample_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/10182018/Fish3-2/test_sample/'  # Currently 20 samples


class PipelineStep(Enum):
    PIXEL_DENOISE   = 1
    REGISTRATION    = 2
    DETREND         = 3
    LOCALPCA        = 4
    DEMIX           = 5

#################################### Utils ########################################
def load_img_seq_dask(img_folder, component):   #filename = imgDMotion for registration and Y_dt for detrend and Y_2dsvd for localPCA
    filename = ''
    concataxis = 0
    if component == PipelineStep.REGISTRATION:
        filename = 'imgDMotion'
    elif component == PipelineStep.DETREND:
        filename = 'Y_dt'
        concataxis = -1
    elif component == PipelineStep.LOCALPCA:
        filename = 'Y_2dsvd'

    imread = dask.delayed(io.imread, pure=True)  # Lazy version of imread
    imgFiles = sorted(glob(img_folder + filename+'*.tif'))
    #start_time = time.time()
    lazy_images = [imread(img).astype('float32') for img in imgFiles]  # Lazily evaluate imread on each path
    sample = lazy_images[0].compute()  # load the first image (assume rest are same shape/dtype)
    arrays = [da.from_delayed(lazy_image,  # Construct a small Dask array
                              dtype=sample.dtype,  # for every lazy value
                              shape=sample.shape)
              for lazy_image in lazy_images]

    imgStack = da.concatenate(arrays, axis=concataxis).astype('float32')
    imgStack = imgStack.compute()
    #print("--- %s seconds for image stack creation: dask ---" % (time.time() - start_time))
    return imgStack



#################################### Regular with multiple load and save #######################################
def demix_middle_data_with_mask_multiple(row, ext=''):
    # to run plt without the X server
    #import matplotlib as mpl
    #mpl.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from skimage.external.tifffile import imsave, imread
    from fish_proc.demix import superpixel_analysis as sup
    from fish_proc.utils.snr import correlation_pnr
    from fish_proc.utils.noise_estimator import get_noise_fft
    import pickle

    save_folder_Registration =  save_folder + 'Registration/'
    save_folder_Detrend = save_folder + 'Detrend/'
    save_folder_LocalPCA = save_folder + 'LocalPCA/divide5block5x5/svdsparse/'
    save_folder_Demix = save_folder + 'Demix/divide5block5x5/svdsparse/'

    sns.set(font_scale=2)
    sns.set_style("white")
    # mask out the region with low snr

    folder = row['folder']
    fish = row['fish']

    save_image_folder = save_folder_Demix + '/Results'
    if not os.path.exists(save_image_folder):
        os.makedirs(save_image_folder)
    print('=====================================')
    print(save_folder)

    if os.path.isfile(save_folder_Demix+f'/finished_demix{ext}.tmp'):
        return None

    if not os.path.isfile(save_folder_Demix+f'/proc_demix{ext}.tmp'):
        Path(save_folder_Demix+f'/proc_demix{ext}.tmp').touch()
        start_time = time.time()
        Y_2dnorm = np.load(f'{save_folder_LocalPCA}/Y_2dnorm.npz')
        print("--- %s seconds for read Y_2dnorm ---" % (time.time() - start_time))

        Y_d_ave= Y_2dnorm['Y_d_ave']
        Y_d_std= Y_2dnorm['Y_d_std']

        start_time = time.time()
        ## TODO: Put this part in the localpca test
        # if not os.path.isfile(f'{save_folder_LocalPCA}/Y_svd.tif'):
        Y_svd = []
        n_splits = 5
        for n_ in range(n_splits):
            Y_2dsvd =  np.load(f'{save_folder_LocalPCA}/Y_2dsvd{n_}.npy').astype('float32') # added for debugging
            Y_svd.append(Y_2dsvd)
        Y_svd = np.concatenate(Y_svd, axis=-1)
        print(Y_svd.shape)
        #imsave(f'{save_folder_LocalPCA}/Y_svd.tif', Y_svd.astype('float32'), compress=1) #remove the save
        print('Concatenate files into a tif file')
        # else:
        #     Y_svd = imread(f'{save_folder_LocalPCA}/Y_svd.tif').astype('float32')

        # multiple load for Y_svd tiff
        #Y_svd = load_img_seq_dask(save_folder_LocalPCA, component=PipelineStep.LOCALPCA)
        print("--- %s seconds for loading Y_svd--" % (time.time() - start_time))

        # comment this for now because I may need the files later
        # start_time = time.time()
        # for n_ in range(10):
        #     if os.path.isfile(f'{save_folder_LocalPCA}/Y_2dsvd{n_}.npy'):
        #         os.remove(f'{save_folder_LocalPCA}/Y_2dsvd{n_}.npy')
        # get_process_memory()
        # print("--- %s seconds for remove partial Y_svd---" % (time.time() - start_time))


        # make mask
        start_time = time.time()
        mean_ = Y_svd.mean(axis=-1,keepdims=True)
        sn, _ = get_noise_fft(Y_svd - mean_,noise_method='logmexp')
        SNR_ = Y_svd.var(axis=-1)/sn**2
        std_thres = np.percentile(Y_d_std.ravel(), 80)
        mask = Y_d_std.squeeze(axis=-1)<=std_thres
        snr_thres = np.percentile(np.log(SNR_).ravel(), 0)
        mask = np.logical_or(mask, np.log(SNR_)<snr_thres)
        mask_out_region = np.logical_not(mask)
        mask_save = np.where(mask_out_region)
        print("--- %s seconds for making mask---" % (time.time() - start_time))

        start_time = time.time()
        np.savez(f'{save_folder_Demix}/mask', mask=mask, mask_save=mask_save)
        print("--- %s seconds for saving mask---" % (time.time() - start_time))

        # get data
        start_time = time.time()
        len_Y = Y_svd.shape[-1]
        mov_ = Y_svd[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max(), len_Y//3:-len_Y//3]
        Y_svd = None
        clear_variables(Y_svd)
        #get_process_memory()
        print("--- %s seconds for get data---" % (time.time() - start_time))

        # get sparse data
        start_time = time.time()
        Y_d_std_ = Y_d_std[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max(), :]
        mov_ = mov_*Y_d_std_ + np.random.normal(size=mov_.shape)*0.7
        mov_ = -mov_.astype('float32')
        mask_ = mask[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max()]
        mov_[mask_]=0
        #get_process_memory()
        d1, d2, _ = mov_.shape
        print("--- %s seconds for get sparse data---" % (time.time() - start_time))

        # get local correlation distribution
        start_time = time.time()
        Cn, _ = correlation_pnr(mov_, skip_pnr=True)
        #get_process_memory()
        print("--- %s seconds for get local correlation distribution---" % (time.time() - start_time))

        start_time = time.time()
        pass_num = 4
        cut_off_point=np.percentile(Cn.ravel(), [99, 95, 85, 65])
        rlt_= sup.demix_whole_data(mov_, cut_off_point, length_cut=[10,15,15,15],
                                   th=[1,1,1,1], pass_num=pass_num, residual_cut = [0.6,0.6,0.6,0.6],
                                   corr_th_fix=0.3, max_allow_neuron_size=0.05, merge_corr_thr=cut_off_point[-1],
                                   merge_overlap_thr=0.6, num_plane=1, patch_size=[40, 40], plot_en=False,
                                   TF=False, fudge_factor=1, text=False, bg=False, max_iter=60,
                                   max_iter_fin=100, update_after=20)
        print("--- %s seconds for demix_whole_data---" % (time.time() - start_time))

        start_time = time.time()
        with open(f'{save_folder_Demix}/period_Y_demix{ext}_rlt.pkl', 'wb') as f:
            pickle.dump(rlt_, f)

        print('Result file saved?')
        print(os.path.isfile(f'{save_folder_Demix}/period_Y_demix{ext}_rlt.pkl'))

        with open(f'{save_folder_Demix}/period_Y_demix{ext}_rlt.pkl', 'rb') as f:
            rlt_ = pickle.load(f)

        print("--- %s seconds for period_Y_demix--" % (time.time() - start_time))


        ##TODO: Potential: save the mean in the registration and detrend step instead of reloading the image to get the mean
        if not os.path.isfile(f'{save_folder_Detrend}/Y_trend_ave.npy'):
            start_time = time.time()
            Y_mean = load_img_seq_dask(save_folder_Registration, component=PipelineStep.REGISTRATION).mean(axis=0)
            #Y_mean = imread(f'{save_folder}/imgDMotion.tif').mean(axis=0)
            Y_d_mean = load_img_seq_dask(save_folder_Detrend, component=PipelineStep.DETREND).mean(axis=-1)
            #Y_d_mean = imread(f'{save_folder}/Y_d.tif').mean(axis=-1)
            print("--- %s seconds for load registration and detrend---" % (time.time() - start_time))

            Y_trend_ave = Y_mean - Y_d_mean
            Y_mean = None
            Y_d_mean = None
            Y_svd_ = None
            clear_variables((Y_d_mean, Y_mean, mov_))
            start_time = time.time()
            np.save(f'{save_folder_Detrend}/Y_trend_ave', Y_trend_ave)
            print("--- %s seconds for save  trend_ave---" % (time.time() - start_time))

        else:
            Y_trend_ave = np.load(f'{save_folder_Detrend}/Y_trend_ave.npy')

        Y_trend_ave = Y_trend_ave[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max()]
        print("--- %s seconds for Y_trend_ave masking---" % (time.time() - start_time))

        start_time = time.time()
        A = rlt_['fin_rlt']['a']
        A_ = A[:, (A>0).sum(axis=0)>0]
        A_comp = np.zeros(A_.shape[0])
        A_comp[A_.sum(axis=-1)>0] = np.argmax(A_[A_.sum(axis=-1)>0, :], axis=-1) + 1
        print("--- %s seconds for component prepartion---" % (time.time() - start_time))

        ### comment the plot part
        start_time = time.time()
        plt.figure(figsize=(8,4))
        plt.imshow(Y_trend_ave, cmap=plt.cm.gray)
        plt.imshow(A_comp.reshape(d2, d1).T, cmap=plt.cm.nipy_spectral_r, alpha=0.7)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_image_folder}/Demixed_components{ext}.png')
        plt.close()
        Path(save_folder_Demix+f'/finished_demix{ext}.tmp').touch()
        print("--- %s seconds forsaving figure---" % (time.time() - start_time))

    return None






#################################### Regular #######################################
def demix_middle_data_with_mask(row, ext=''):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from skimage.external.tifffile import imsave, imread
    from fish_proc.demix import superpixel_analysis as sup
    from fish_proc.utils.snr import correlation_pnr
    from fish_proc.utils.noise_estimator import get_noise_fft
    import pickle
    sns.set(font_scale=2)
    sns.set_style("white")
    # mask out the region with low snr

    folder = row['folder']
    fish = row['fish']

    #save_image_folder = dat_folder + f'{folder}/{fish}/Results'
    save_image_folder = save_folder + 'Results'
    if not os.path.exists(save_image_folder):
        os.makedirs(save_image_folder)
    print('=====================================')
    print(save_folder)

    #if os.path.isfile(save_folder+f'/finished_demix{ext}.tmp'):
    #    return None

    #if not os.path.isfile(save_folder+f'/proc_demix{ext}.tmp'):
    Path(save_folder+f'/proc_demix{ext}.tmp').touch()
    _ = np.load(f'{save_folder}/Y_2dnorm.npz')
    Y_d_ave= _['Y_d_ave']
    Y_d_std= _['Y_d_std']

    start_time = time.time()
    if not os.path.isfile(f'{save_folder}/Y_svd.tif'):
        Y_svd = []
        for n_ in range(10):
            Y_2dsvd = np.load(f'{save_folder}/Y_2dsvd{n_}.npy').astype('float32') # added for debugging
            Y_svd.append(Y_2dsvd)
        Y_svd = np.concatenate(Y_svd, axis=-1)
        print(Y_svd.shape)
        imsave(f'{save_folder}/Y_svd.tif', Y_svd.astype('float32'), compress=1)
        print('Concatenate files into a tif file')
    else: # just for testing
        Y_svd = imread(f'{save_folder}/Y_svd.tif').astype('float32')
    print("--- %s seconds for read Y_svd---" % (time.time() - start_time))

    print("Y_svd shape")
    print(Y_svd.shape)

    # start_time = time.time()
    # for n_ in range(10):
    #     if os.path.isfile(f'{save_folder}/Y_2dsvd{n_}.npy'):
    #         os.remove(f'{save_folder}/Y_2dsvd{n_}.npy')
    # get_process_memory()
    # print("--- %s seconds for remove Y_svd---" % (time.time() - start_time))


    # make mask
    start_time = time.time()
    mean_ = Y_svd.mean(axis=-1,keepdims=True)
    sn, _ = get_noise_fft(Y_svd - mean_,noise_method='logmexp')
    SNR_ = Y_svd.var(axis=-1)/sn**2
    std_thres = np.percentile(Y_d_std.ravel(), 80)
    mask = Y_d_std.squeeze(axis=-1)<=std_thres
    snr_thres = np.percentile(np.log(SNR_).ravel(), 0)
    mask = np.logical_or(mask, np.log(SNR_)<snr_thres)
    mask_out_region = np.logical_not(mask)
    mask_save = np.where(mask_out_region)
    print("--- %s seconds for making mask---" % (time.time() - start_time))

    start_time = time.time()
    np.savez(f'{save_folder}/mask', mask=mask, mask_save=mask_save)
    print("--- %s seconds for saving mask---" % (time.time() - start_time))

    # get data
    start_time = time.time()
    len_Y = Y_svd.shape[-1]
    mov_ = Y_svd[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max(), len_Y//3:-len_Y//3]
    Y_svd = None
    clear_variables(Y_svd)
    #get_process_memory()
    print("--- %s seconds for get data---" % (time.time() - start_time))

    # get sparse data
    start_time = time.time()
    Y_d_std_ = Y_d_std[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max(), :]

    mov_ = mov_*Y_d_std_ + np.random.normal(size=mov_.shape)*0.7
    mov_ = -mov_.astype('float32')
    mask_ = mask[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max()]
    mov_[mask_]=0
    #get_process_memory()
    d1, d2, _ = mov_.shape
    print("--- %s seconds for get sparse data---" % (time.time() - start_time))

    # get local correlation distribution
    start_time = time.time()
    Cn, _ = correlation_pnr(mov_, skip_pnr=True)
    #get_process_memory()
    print("--- %s seconds for get local correlation distribution---" % (time.time() - start_time))

    start_time = time.time()
    pass_num = 4
    cut_off_point=np.percentile(Cn.ravel(), [99, 95, 85, 65])
    rlt_= sup.demix_whole_data(mov_, cut_off_point, length_cut=[10,15,15,15],
                               th=[1,1,1,1], pass_num=pass_num, residual_cut = [0.6,0.6,0.6,0.6],
                               corr_th_fix=0.3, max_allow_neuron_size=0.05, merge_corr_thr=cut_off_point[-1],
                               merge_overlap_thr=0.6, num_plane=1, patch_size=[40, 40], plot_en=False,
                               TF=False, fudge_factor=1, text=False, bg=False, max_iter=60,
                               max_iter_fin=100, update_after=20)
    print("--- %s seconds for demix_whole_data---" % (time.time() - start_time))

    start_time = time.time()
    with open(f'{save_folder}/period_Y_demix{ext}_rlt.pkl', 'wb') as f:
        pickle.dump(rlt_, f)

    print('Result file saved?')
    print(os.path.isfile(f'{save_folder}/period_Y_demix{ext}_rlt.pkl'))

    with open(f'{save_folder}/period_Y_demix{ext}_rlt.pkl', 'rb') as f:
        rlt_ = pickle.load(f)

    if not os.path.isfile(f'{save_folder}/Y_trend_ave.npy'):
        Y_mean = imread(f'{save_folder}/imgDMotion.tif').mean(axis=0)
        Y_d_mean = imread(f'{save_folder}/Y_d.tif').mean(axis=-1)
        Y_trend_ave = Y_mean - Y_d_mean
        Y_mean = None
        Y_d_mean = None
        Y_svd_ = None
        clear_variables((Y_d_mean, Y_mean, mov_))
        np.save(f'{save_folder}/Y_trend_ave', Y_trend_ave)
    else:
        Y_trend_ave = np.load(f'{save_folder}/Y_trend_ave.npy')

    Y_trend_ave = Y_trend_ave[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max()]
    print("--- %s seconds for Y_trend_ave---" % (time.time() - start_time))

    start_time = time.time()
    A = rlt_['fin_rlt']['a']
    A_ = A[:, (A>0).sum(axis=0)>0]
    A_comp = np.zeros(A_.shape[0])
    A_comp[A_.sum(axis=-1)>0] = np.argmax(A_[A_.sum(axis=-1)>0, :], axis=-1) + 1
    print("--- %s seconds for component prepartion---" % (time.time() - start_time))

    ### comment the plot part
    plt.figure(figsize=(8,4))
    plt.imshow(Y_trend_ave, cmap=plt.cm.gray)
    plt.imshow(A_comp.reshape(d2, d1).T, cmap=plt.cm.nipy_spectral_r, alpha=0.7)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_image_folder}/Demixed_components{ext}.png')
    plt.close()
    Path(save_folder+f'/finished_demix{ext}.tmp').touch()
    return None



def demix_components(ext=''):
    dat_xls_file = pd.read_csv('./Voltron_Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
    for index, row in dat_xls_file.iterrows():
        #demix_middle_data_with_mask(row, ext=ext)
        demix_middle_data_with_mask_multiple(row, ext=ext)
    return None


######### Tests and main #########

## python Dask_registration_test.py
if __name__ == '__main__':
    # fix the random seed
    import random
    random.seed(0)
    if len(sys.argv)>1:
        ext = ''
        if len(sys.argv)>2:
            ext = sys.argv[2]
        eval(sys.argv[1]+f"({ext})")
    else:
        start_time = time.time()
        demix_components()
        print("--- %s seconds for demix---" % (time.time() - start_time))
