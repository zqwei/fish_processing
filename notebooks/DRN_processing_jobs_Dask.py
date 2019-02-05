
"""
Using dask

@author: modified by Salma Elmalaki, Dec 2018

"""

import numpy as np
import pandas as pd
import os, sys
from fish_proc.utils.memory import get_process_memory, clear_variables
import time
from pathlib import Path
import dask
from glob import glob
import dask.array as da
from skimage import io
from enum import Enum

dat_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/ProcessedData/'
cameraNoiseMat = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/gainMat/gainMat20180208'


#################################### Utils ########################################
class PipelineStep(Enum):
    PIXEL_DENOISE   = 1
    REGISTRATION    = 2
    DETREND         = 3
    LOCALPCA        = 4
    DEMIX           = 5


def load_img_seq_dask(img_folder, component):
    filename = ''
    concataxis = 0
    if component == PipelineStep.PIXEL_DENOISE:
        filename= 'imgDNoMotion'
    elif component == PipelineStep.REGISTRATION:
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


def save_img_seq_dask(save_img_folder, img_splits, component):
    filename = ''
    if component == PipelineStep.PIXEL_DENOISE:
        filename= 'imgDNoMotion'
    elif component == PipelineStep.REGISTRATION:
        filename = 'imgDMotion'
    elif component == PipelineStep.DETREND:
        filename = 'Y_dt'
    elif component == PipelineStep.LOCALPCA:
        filename = 'Y_2dsvd'

    delayed_imsave = dask.delayed(io.imsave, pure=True)  # Lazy version of imsave
    lazy_images = [
        delayed_imsave(save_img_folder + '/' + filename + '%04d' % index + '.tif', img, compress=1)
        for index, img in enumerate(img_splits)
                    ]     # Lazily evaluate imsave on each path
    dask.compute(*lazy_images)

#################################### ########################################

def monitor_process():
    '''
    Update Voltron Log_DRN_Exp.csv
    monitor process of processing
    '''
    dat_xls_file = pd.read_csv('./Voltron_Log_DRN_Exp.csv', index_col=0)
    if 'index' in dat_xls_file.columns:
        dat_xls_file = dat_xls_file.drop('index', axis=1)
    dat_xls_file['folder'] = dat_xls_file['folder'].astype(int).apply(str)

    if not 'spikes' in dat_xls_file.columns:
        dat_xls_file['spikes']=False
    if not 'subvolt' in dat_xls_file.columns:
        dat_xls_file['subvolt']=False

    dat_xls_file['folder'] = dat_xls_file['folder'].astype(int).apply(str)
    for index, row in dat_xls_file.iterrows():
        # check swim:
        folder = row['folder']
        fish = row['fish']
        save_folder = dat_folder + f'{folder}/{fish}/'
        if os.path.exists(save_folder+'/swim'):
            dat_xls_file.at[index, 'swim'] = True
        if os.path.isfile(save_folder + '/Data/motion_fix_.npy'):
            dat_xls_file.at[index, 'pixeldenoise'] = True
        if os.path.isfile(save_folder+'/Data/finished_registr.tmp'):
            dat_xls_file.at[index, 'registration'] = True
        if os.path.isfile(save_folder+'/Data/finished_detrend.tmp'):
            dat_xls_file.at[index, 'detrend'] = True
        if os.path.isfile(save_folder+'/Data/finished_local_denoise.tmp'):
            dat_xls_file.at[index, 'localdenoise'] = True
        if os.path.isfile(save_folder+'/Data/finished_demix.tmp'):
            dat_xls_file.at[index, 'demix'] = True
        if os.path.isfile(save_folder+'/Data/finished_voltr.tmp'):
            dat_xls_file.at[index, 'voltr'] = True
        if os.path.exists(save_folder+'/Data/finished_spikes.tmp'):
            dat_xls_file.at[index, 'spikes'] = True
        if os.path.exists(save_folder+'/Data/finished_subvolt.tmp'):
            dat_xls_file.at[index, 'subvolt'] = True
    print(dat_xls_file.sum(numeric_only=True))
    dat_xls_file.to_csv('./Voltron_Log_DRN_Exp.csv')
    return None


def swim():
    '''
    Processing swim using TK's code
    '''
    from fish_proc.utils.ep import process_swim

    dat_xls_file = pd.read_csv('./Voltron_Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
    for _, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        swim_chFit = f'/nrs/ahrens/Takashi/{folder}/{fish}.10chFlt'
        save_folder = dat_folder + f'{folder}/{fish}/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(save_folder+'/swim'):
            print(f'checking file {folder}/{fish}')
            try:
                process_swim(swim_chFit, save_folder)
            except IOError:
                print(f'Check existence of file {swim_chFit}')
    return None



def pixel_denoise():
    '''
    Process pixel denoise
    Generate files -- imgDNoMotion.tif, motion_fix_.npy
    '''
    from fish_proc.pipeline.preprocess_dask import pixel_denoise, pixel_denoise_img_seq
    import time
    import dask.array as da

    start_time_init = time.time()
    dat_xls_file = pd.read_csv('./Voltron_Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        image_folder = row['rootDir'] + f'{folder}/{fish}/'
        fish_folder = dat_folder + f'{folder}/{fish}/'
        save_folder = dat_folder + f'{folder}/{fish}/Data/'

        save_folder_PixelDenoise = save_folder + 'PixelDenoise/'

        if os.path.exists(image_folder):
            print(f'checking file {folder}/{fish}')
            if not os.path.exists(save_folder_PixelDenoise):
                os.makedirs(save_folder_PixelDenoise)
            if os.path.exists(save_folder_PixelDenoise + 'finished_pixel_denoise.tmp'):
                continue
            if not os.path.isfile(save_folder_PixelDenoise + 'motion_fix.npy'):
                print(f'process file {folder}/{fish}')
                try:
                    if os.path.exists(image_folder+'Registered/raw.tif'):
                        imgD_ = pixel_denoise(image_folder, 'Registered/raw.tif', save_folder_PixelDenoise, cameraNoiseMat, plot_en=False)
                    else:
                        start_time = time.time()
                        imgD_ = pixel_denoise_img_seq(image_folder, save_folder_PixelDenoise, cameraNoiseMat, plot_en=False)
                        print("--- %s seconds for pixel denoising function ---" % (time.time() - start_time))

                    t_ = len(imgD_)//2
                    win_ = 150

                    start_time = time.time()
                    fix_ = imgD_[t_ - win_:t_ + win_].mean(axis=0)
                    print("--- %s seconds for fix mean ---" % (time.time() - start_time))

                    start_time = time.time()
                    np.save(save_folder_PixelDenoise + '/motion_fix', fix_)
                    print("--- %s seconds for saving motionfix ---" % (time.time() - start_time))

                    #get_process_memory()
                    imgD_ = None
                    fix_ = None
                    clear_variables((imgD_, fix_))
                    #print("--- %s seconds for saving motion_fix and clearning variables ---" % (time.time() - start_time))
                    Path(save_folder_PixelDenoise + '/finished_pixel_denoise.tmp').touch()

                except MemoryError as err:
                    print(f'Memory Error on file {folder}/{fish}: {err}')
    print("--- %s seconds for pixel denoise ---" % (time.time() - start_time_init))
    return None




def registration(is_largefile=True):
    '''
    Generate imgDMotion.tif
    '''
    from pathlib import Path
    from fish_proc.pipeline.preprocess import motion_correction
    from skimage.io import imread, imsave
    import time

    dat_xls_file = pd.read_csv('./Voltron_Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        save_folder = dat_folder + f'{folder}/{fish}/Data/'

        save_folder_Registration = save_folder + 'Registration/'
        save_folder_PixelDenoise = save_folder + 'PixelDenoise/'

        print(f'checking file {folder}/{fish}')
        if os.path.isfile(save_folder_PixelDenoise+'/finished_pixel_denoise.tmp') and os.path.isfile(save_folder_PixelDenoise + '/motion_fix.npy'):
            print('motion fix is found')
            if not os.path.exists(save_folder_Registration):
                os.makedirs(save_folder_Registration)
            if not os.path.isfile(save_folder_Registration+'/proc_registr.tmp'):
                Path(save_folder_Registration+'/proc_registr.tmp').touch()
                print(f'process file {folder}/{fish}')

                start_time = time.time()
                imgD_ = load_img_seq_dask(save_folder_PixelDenoise, component=PipelineStep.PIXEL_DENOISE)
                print("--- %s seconds for loading ---" % (time.time() - start_time))

                start_time = time.time()
                fix_ = np.load(save_folder_PixelDenoise + '/motion_fix.npy').astype('float32')
                print("--- %s seconds for loading motion_fix_.tiff ---" % (time.time() - start_time))

                if is_largefile:
                    len_D_ = len(imgD_)//2
                    motion_correction(imgD_[:len_D_], fix_, save_folder_Registration, ext='0')
                    #get_process_memory();
                    motion_correction(imgD_[len_D_:], fix_, save_folder_Registration, ext='1')
                    #get_process_memory();
                    imgD_ = None
                    fix_ = None
                    clear_variables((imgD_, fix_))

                    start_time = time.time()
                    s_ = [np.load(save_folder_Registration+'/imgDMotion%d.npy'%(__)) for __ in range(2)]
                    s_ = np.concatenate(s_, axis=0).astype('float32')
                    print("--- %s seconds for loading load and concatenate imgDMotion ---" % (time.time() - start_time))

                    start_time = time.time()  # --salma
                    n_splits = s_.shape[0] // 50
                    imgSplit = np.split(s_, n_splits)
                    save_img_seq_dask(save_folder_Registration, imgSplit, component=PipelineStep.REGISTRATION)
                    print("--- %s seconds for save dask ---" % (time.time() - start_time))

                    s_ = None
                    clear_variables(s_)
                    os.remove(save_folder_Registration+'/imgDMotion0.npy')
                    os.remove(save_folder_Registration+'/imgDMotion1.npy')
                else:
                    motion_correction(imgD_, fix_, save_folder_Registration)
                    #get_process_memory();
                    imgD_ = None
                    fix_ = None
                    clear_variables((imgD_, fix_))
                    s_ = np.load(save_folder_Registration+'/imgDMotion.npy').astype('float32')
                    imsave(save_folder_Registration+'/imgDMotion.tif', s_, compress=1)
                    s_ = None
                    clear_variables(s_)
                    os.remove(save_folder_Registration+'/imgDMotion.npy')
                Path(save_folder_Registration+'/finished_registr.tmp').touch()

    return None



def video_detrend():
    from fish_proc.pipeline.denoise import detrend
    from pathlib import Path
    from multiprocessing import cpu_count
    from skimage.io import imsave, imread
    dat_xls_file = pd.read_csv('./Voltron_Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        save_folder = dat_folder + f'{folder}/{fish}/Data/'

        save_folder_Registration = save_folder + 'Registration/'
        save_folder_Detrend = save_folder + 'Detrend/'

        print(f'checking file {folder}/{fish}')
        if not os.path.exists(save_folder_Detrend):
            os.makedirs(save_folder_Detrend)
        if os.path.isfile(save_folder_Detrend+'/finished_detrend.tmp'):
            continue

        if not os.path.isfile(save_folder_Detrend+'/proc_detrend.tmp'):
            if os.path.isfile(save_folder_Registration+'/finished_registr.tmp'):
                Path(save_folder_Detrend+'/proc_detrend.tmp').touch()

                start_time = time.time()
                Y = load_img_seq_dask(save_folder_Registration, component=PipelineStep.REGISTRATION)
                print("--- %s seconds for read---" % (time.time() - start_time))

                start_time = time.time()
                Y = Y.transpose([1,2,0])
                print("--- %s seconds for transpose---" % (time.time() - start_time))

                n_split = min(Y.shape[0]//cpu_count(), 8)
                if n_split <= 1:
                    n_split = 2
                Y_len = Y.shape[0]//2
                start_time = time.time()
                detrend(Y[:Y_len, :, :], save_folder_Detrend, n_split=n_split//2, ext='0')
                print("--- %s seconds for detrend1---" % (time.time() - start_time))

                start_time = time.time()
                detrend(Y[Y_len:, :, :], save_folder_Detrend, n_split=n_split//2, ext='1')
                print("--- %s seconds for detrend2---" % (time.time() - start_time))

                Y = None
                clear_variables(Y)
                get_process_memory()
                Y = []
                start_time = time.time()
                Y.append(np.load(save_folder_Detrend+'/Y_d0.npy').astype('float32'))
                Y.append(np.load(save_folder_Detrend+'/Y_d1.npy').astype('float32'))
                print("--- %s seconds for append---" % (time.time() - start_time))

                start_time = time.time()
                Y = np.concatenate(Y, axis=0).astype('float32')
                print("--- %s seconds for concat---" % (time.time() - start_time))
                # Save the npy. It is needed for localPCA
                np.save(f'{save_folder_Detrend}/Y_d.npy', Y)

                # multiple save
                n_splits = Y.shape[-1] // 50
                imgSplit = np.split(Y, n_splits, axis=2)
                save_img_seq_dask(save_folder_Detrend, imgSplit, component=PipelineStep.DETREND)
                print("--- %s seconds for save  ---" % (time.time() - start_time))  # --salma

                Y = None
                clear_variables(Y)
                get_process_memory()
                os.remove(save_folder_Detrend+'/Y_d0.npy')
                os.remove(save_folder_Detrend+'/Y_d1.npy')
                Path(save_folder_Detrend+'/finished_detrend.tmp').touch()
    return None


def local_pca():
    from fish_proc.pipeline.denoise import denose_2dsvd
    from pathlib import Path
    from skimage.external.tifffile import imsave, imread
    dat_xls_file = pd.read_csv('./Voltron_Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        image_folder = f'/nrs/ahrens/Takashi/0{folder}/{fish}/'
        save_folder = dat_folder + f'{folder}/{fish}/Data/'

        save_folder_Detrend = save_folder + 'Detrend/'
        save_folder_LocalPCA = save_folder + 'LocalPCA/'

        print(f'checking file {folder}/{fish}')

        if not os.path.exists(save_folder_LocalPCA):
            os.makedirs(save_folder_LocalPCA)

        if os.path.isfile(save_folder_LocalPCA+'/finished_local_denoise.tmp'):
            continue

        if not os.path.isfile(save_folder_LocalPCA+'/proc_local_denoise.tmp'):
            if os.path.isfile(save_folder_Detrend+'/finished_detrend.tmp'):
                Path(save_folder_LocalPCA+'/proc_local_denoise.tmp').touch()

                if os.path.isfile(f'{save_folder_Detrend}/Y_d.npy'):
                    Y_d = np.load(f'{save_folder_Detrend}/Y_d.npy').astype('float32')
                else:
                    Y_d = load_img_seq_dask(save_folder_Detrend, component=PipelineStep.DETREND)

                Y_d_ave = Y_d.mean(axis=-1, keepdims=True) # remove mean
                Y_d_std = Y_d.std(axis=-1, keepdims=True) # normalization
                Y_d = (Y_d - Y_d_ave)/Y_d_std
                Y_d = Y_d.astype('float32')
                np.savez_compressed(f'{save_folder_LocalPCA}/Y_2dnorm', Y_d_ave=Y_d_ave, Y_d_std=Y_d_std)
                Y_d_ave = None
                Y_d_std = None
                clear_variables((Y_d_ave, Y_d_std))
                get_process_memory()

                n_splits = 10
                nblocks = [10, 10]
                for n, Y_d_ in enumerate(np.array_split(Y_d, n_splits, axis=-1)):
                    denose_2dsvd(Y_d_, save_folder_LocalPCA, nblocks, ext=f'{n}')

                Y_d_ = None
                Y_d = None
                clear_variables(Y_d)
                get_process_memory()
                Path(save_folder_LocalPCA+'/finished_local_denoise.tmp').touch()
    return None


def demix_middle_data():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from skimage.external.tifffile import imsave, imread
    from fish_proc.demix import superpixel_analysis as sup
    from fish_proc.utils.snr import correlation_pnr
    import pickle

    sns.set(font_scale=2)
    sns.set_style("white")
    dat_xls_file = pd.read_csv('./Voltron_Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        image_folder = f'/nrs/ahrens/Takashi/0{folder}/{fish}/'
        save_folder = dat_folder + f'{folder}/{fish}/Data/'
        save_image_folder = dat_folder + f'{folder}/{fish}/Results'

        if not os.path.exists(save_image_folder):
            os.makedirs(save_image_folder)
        print('=====================================')
        print(save_folder)

        if os.path.isfile(save_folder+'/finished_demix.tmp'):
            continue

        if not os.path.isfile(save_folder+'/proc_demix.tmp'):
            Path(save_folder+'/proc_demix.tmp').touch()
            _ = np.load(f'{save_folder}/Y_2dnorm.npz')
            Y_d_ave= _['Y_d_ave']
            Y_d_std= _['Y_d_std']
            if not os.path.isfile(f'{save_folder}/Y_svd.tif'):
                Y_svd = []
                for n_ in range(10):
                    Y_svd.append(np.load(f'{save_folder}/Y_2dsvd{n_}.npy').astype('float32'))
                Y_svd = np.concatenate(Y_svd, axis=-1)
                print(Y_svd.shape)
                imsave(f'{save_folder}/Y_svd.tif', Y_svd.astype('float32'), compress=1)
                print('Concatenate files into a tif file')
            else:
                Y_svd = imread(f'{save_folder}/Y_svd.tif').astype('float32')

            for n_ in range(10):
                if os.path.isfile(f'{save_folder}/Y_2dsvd{n_}.npy'):
                    os.remove(f'{save_folder}/Y_2dsvd{n_}.npy')
            get_process_memory();

            plt.figure(figsize=(8, 4))
            plt.imshow(Y_d_std[:, :, 0])
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f'{save_image_folder}/Detrend_std.png')
            plt.close()

            len_Y = Y_svd.shape[-1]
            Y_svd_ = Y_svd[:, :, len_Y//4:-len_Y//4]
            d1, d2, _ = Y_svd_.shape
            Y_svd = None
            clear_variables(Y_svd)
            get_process_memory();

            np.random.seed(0); mov_ = Y_svd_*Y_d_std + np.random.normal(size=Y_svd_.shape)*0.7
            Y_svd_ = None
            clear_variables(Y_svd_)
            Cn, _ = correlation_pnr(-mov_, gSig=None, remove_small_val=False, remove_small_val_th=3, center_psf=False)
            plt.figure(figsize=(8, 4))
            plt.imshow(Cn, vmin=0.0)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f'{save_image_folder}/Denoised_local_corr_noised.png')
            plt.close()

            pass_num = 4
            cut_off_point=np.percentile(Cn.ravel(), [99, 95, 90, 85])
            rlt_= sup.demix_whole_data(-mov_, cut_off_point, length_cut=[25,25,25,25],
                                       th=[2,2,1,1], pass_num=pass_num, residual_cut = [0.6,0.6,0.6,0.6],
                                       corr_th_fix=0.3, max_allow_neuron_size=0.2, merge_corr_thr=0.35,
                                       merge_overlap_thr=0.6, num_plane=1, patch_size=[100, 100], plot_en=False,
                                       TF=False, fudge_factor=1, text=False, bg=False, max_iter=60,
                                       max_iter_fin=100, update_after=4)

            with open(f'{save_folder}/period_Y_demix_rlt.pkl', 'wb') as f:
                pickle.dump(rlt_, f)

            print('Result file saved?')
            print(os.path.isfile(f'{save_folder}/period_Y_demix_rlt.pkl'))

            with open(f'{save_folder}/period_Y_demix_rlt.pkl', 'rb') as f:
                rlt_ = pickle.load(f)

            for n_pass in range(pass_num):
                sup.pure_superpixel_single_plot(rlt_["superpixel_rlt"][n_pass]["connect_mat_1"],
                                                rlt_["superpixel_rlt"][n_pass]["unique_pix"],
                                                rlt_["superpixel_rlt"][n_pass]["brightness_rank_sup"],
                                                text=True,
                                                pure=False);
                plt.savefig(f'{save_image_folder}/Demixed_pass_#{n_pass}.png')
                plt.close()

            Y_mean = imread(f'{save_folder}/imgDMotion.tif').mean(axis=0)
            Y_d_mean = imread(f'{save_folder}/Y_d.tif').mean(axis=-1)
            Y_trend_ave = Y_mean - Y_d_mean
            Y_mean = None
            Y_d_mean = None
            Y_svd_ = None
            clear_variables((Y_d_mean, Y_mean, mov_))

            np.save(f'{save_folder}/Y_trend_ave', Y_trend_ave)

            A = rlt_['fin_rlt']['a']
            A_ = A[:, (A>0).sum(axis=0)>0]
            A_comp = np.zeros(A_.shape[0])
            A_comp[A_.sum(axis=-1)>0] = np.argmax(A_[A_.sum(axis=-1)>0, :], axis=-1) + 1
            plt.figure(figsize=(8,4))
            plt.imshow(Y_trend_ave, cmap=plt.cm.gray)
            plt.imshow(A_comp.reshape(d2, d1).T, cmap=plt.cm.nipy_spectral_r, alpha=0.7)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_image_folder}/Demixed_components.png')
            plt.close()
            Path(save_folder+'/finished_demix.tmp').touch()
    return None


def demix_components(ext=''):
    dat_xls_file = pd.read_csv('./Voltron_Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
    for index, row in dat_xls_file.iterrows():
        demix_middle_data_with_mask(row, ext=ext)
    return None


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
    image_folder = f'/nrs/ahrens/Takashi/{folder}/{fish}/'
    save_folder = dat_folder + f'{folder}/{fish}/Data/'
    save_image_folder = dat_folder + f'{folder}/{fish}/Results'

    save_folder_Registration = save_folder + 'Registration/'
    save_folder_Detrend = save_folder + 'Detrend/'
    save_folder_LocalPCA = save_folder + 'LocalPCA/'
    save_folder_Demix = save_folder + 'Demix/'


    if not os.path.exists(save_image_folder):
        os.makedirs(save_image_folder)
    print('=====================================')
    print(save_image_folder)

    if not os.path.exists(save_folder_Demix):
        os.makedirs(save_folder_Demix)

    if os.path.isfile(save_folder_Demix+f'/finished_demix{ext}.tmp'):
        return None

    if not os.path.isfile(save_folder_Demix+f'/proc_demix{ext}.tmp'):
        Path(save_folder_Demix+f'/proc_demix{ext}.tmp').touch()
        _ = np.load(f'{save_folder_LocalPCA}/Y_2dnorm.npz')
        Y_d_ave= _['Y_d_ave']
        Y_d_std= _['Y_d_std']
        # No need to generate the Y_svd.tif and leave the np version of it
        # if not os.path.isfile(f'{save_folder_LocalPCA}/Y_svd.tif'):
        #     Y_svd = []
        #     for n_ in range(10):
        #         Y_svd.append(np.load(f'{save_folder}/Y_2dsvd{n_}.npy').astype('float32'))
        #     Y_svd = np.concatenate(Y_svd, axis=-1)
        #     print(Y_svd.shape)
        #     imsave(f'{save_folder}/Y_svd.tif', Y_svd.astype('float32'), compress=1)
        #     print('Concatenate files into a tif file')
        # else:
        #     Y_svd = imread(f'{save_folder}/Y_svd.tif').astype('float32')
        #
        # for n_ in range(10):
        #     if os.path.isfile(f'{save_folder}/Y_2dsvd{n_}.npy'):
        #         os.remove(f'{save_folder}/Y_2dsvd{n_}.npy')
        #get_process_memory()

        Y_svd = []
        n_splits = 10
        for n_ in range(n_splits):
            Y_2dsvd =  np.load(f'{save_folder_LocalPCA}/Y_2dsvd{n_}.npy').astype('float32')
            Y_svd.append(Y_2dsvd)
        Y_svd = np.concatenate(Y_svd, axis=-1)

        # make mask
        mean_ = Y_svd.mean(axis=-1,keepdims=True)
        sn, _ = get_noise_fft(Y_svd - mean_,noise_method='logmexp')
        SNR_ = Y_svd.var(axis=-1)/sn**2
        std_thres = np.percentile(Y_d_std.ravel(), 80)
        mask = Y_d_std.squeeze(axis=-1)<=std_thres
        snr_thres = np.percentile(np.log(SNR_).ravel(), 0)
        mask = np.logical_or(mask, np.log(SNR_)<snr_thres)
        mask_out_region = np.logical_not(mask)
        mask_save = np.where(mask_out_region)
        np.savez(f'{save_folder}/mask', mask=mask, mask_save=mask_save)

        # get data
        len_Y = Y_svd.shape[-1]
        mov_ = Y_svd[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max(), len_Y//3:-len_Y//3]
        Y_svd = None
        clear_variables(Y_svd)
        #get_process_memory()

        # get sparse data
        Y_d_std_ = Y_d_std[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max(), :]
        np.random.seed(0); mov_ = mov_*Y_d_std_ + np.random.normal(size=mov_.shape)*0.7
        mov_ = -mov_.astype('float32')
        mask_ = mask[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max()]
        mov_[mask_]=0
        #get_process_memory()
        d1, d2, _ = mov_.shape

        # get local correlation distribution
        Cn, _ = correlation_pnr(mov_, skip_pnr=True)
        #get_process_memory()

        pass_num = 4
        cut_off_point=np.percentile(Cn.ravel(), [99, 95, 85, 65])
        rlt_= sup.demix_whole_data(mov_, cut_off_point, length_cut=[10,15,15,15],
                                   th=[1,1,1,1], pass_num=pass_num, residual_cut = [0.6,0.6,0.6,0.6],
                                   corr_th_fix=0.3, max_allow_neuron_size=0.05, merge_corr_thr=cut_off_point[-1],
                                   merge_overlap_thr=0.6, num_plane=1, patch_size=[40, 40], plot_en=False,
                                   TF=False, fudge_factor=1, text=False, bg=False, max_iter=60,
                                   max_iter_fin=100, update_after=20)

        with open(f'{save_folder_Demix}/period_Y_demix{ext}_rlt.pkl', 'wb') as f:
            pickle.dump(rlt_, f)

        print('Result file saved?')
        print(os.path.isfile(f'{save_folder_Demix}/period_Y_demix{ext}_rlt.pkl'))

        with open(f'{save_folder_Demix}/period_Y_demix{ext}_rlt.pkl', 'rb') as f:
            rlt_ = pickle.load(f)


        if not os.path.isfile(f'{save_folder_Detrend}/Y_trend_ave.npy'):
            start_time = time.time()
            Y_mean = load_img_seq_dask(save_folder_Registration, component=PipelineStep.REGISTRATION).mean(axis=0)
            Y_d_mean = load_img_seq_dask(save_folder_Detrend, component=PipelineStep.DETREND).mean(axis=-1)
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

        A = rlt_['fin_rlt']['a']
        A_ = A[:, (A>0).sum(axis=0)>0]
        A_comp = np.zeros(A_.shape[0])
        A_comp[A_.sum(axis=-1)>0] = np.argmax(A_[A_.sum(axis=-1)>0, :], axis=-1) + 1
        plt.figure(figsize=(8,4))
        plt.imshow(Y_trend_ave, cmap=plt.cm.gray)
        plt.imshow(A_comp.reshape(d2, d1).T, cmap=plt.cm.nipy_spectral_r, alpha=0.7)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_image_folder}/Demixed_components{ext}.png')
        plt.close()
        Path(save_folder_Demix+f'/finished_demix{ext}.tmp').touch()
    return None


if __name__ == '__main__':
    if len(sys.argv)>1:
        ext = ''
        if len(sys.argv)>2:
            ext = sys.argv[2]
        eval(sys.argv[1]+f"({ext})")
    else:
        #monitor_process()
        pixel_denoise()
        #registration()
        #video_detrend()
