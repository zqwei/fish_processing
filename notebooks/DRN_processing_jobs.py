import numpy as np
import pandas as pd
import os, sys
from fish_proc.utils.memory import get_process_memory, clear_variables

dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/'
cameraNoiseMat = '/groups/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'


def update_table(update_ods = False):
    if update_ods:
        dat_xls_file = pd.read_excel(dat_folder+'Voltron Log_DRN_Exp.xlsx')
        dat_xls_file = dat_xls_file.dropna(how='all').reset_index()
        dat_xls_file['folder'] = dat_xls_file['folder'].astype('int').astype('str')
        dat_xls_file['finished'] = False
        dat_xls_file.to_csv(dat_folder + 'Voltron Log_DRN_Exp.csv')
    return None


def monitor_process():
    dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
    if 'index' in dat_xls_file.columns:
        dat_xls_file = dat_xls_file.drop('index', axis=1)
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
    print(dat_xls_file.sum(numeric_only=True))
    # print(dat_xls_file[dat_xls_file['registration']==False])
    dat_xls_file.to_csv(dat_folder + 'Voltron Log_DRN_Exp.csv')
    # save a local copy
    dat_xls_file.to_csv('Voltron Log_DRN_Exp.csv')
    return None

def swim():
    from fish_proc.utils.ep import process_swim
    dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
    for _, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        swim_chFit = f'/nrs/ahrens/Takashi/0{folder}/{fish}.10chFlt'
        save_folder = dat_folder + f'{folder}/{fish}/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(save_folder+'/swim'):
            try:
                process_swim(swim_chFit, save_folder)
            except IOError:
                print(f'Check existence of file {swim_chFit}')
    return None


def pixel_denoise():
    from fish_proc.pipeline.preprocess import pixel_denoise, pixel_denoise_img_seq
    dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
    for _, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        image_folder = f'/nrs/ahrens/Takashi/0{folder}/{fish}/'
        fish_folder = dat_folder + f'{folder}/{fish}/'
        save_folder = dat_folder + f'{folder}/{fish}/Data'
        if os.path.exists(fish_folder):
            print(f'checking file {folder}/{fish}')
            if not os.path.exists(save_folder+'/'):
                os.makedirs(save_folder)
            if os.path.exists(save_folder + 'imgDNoMotion.tif'):
                continue
            if not os.path.isfile(save_folder + '/motion_fix_.npy'):
                print(f'process file {folder}/{fish}')
                try:
                    if os.path.exists(image_folder+'Registered/raw.tif'):
                        imgD_ = pixel_denoise(image_folder, 'Registered/raw.tif', save_folder, cameraNoiseMat, plot_en=True)
                    else:
                        imgD_ = pixel_denoise_img_seq(image_folder, save_folder, cameraNoiseMat, plot_en=True)
                    t_ = len(imgD_)//2
                    win_ = 150
                    fix_ = imgD_[t_-win_:t_+win_].mean(axis=0)
                    np.save(save_folder + '/motion_fix_', fix_)
                    get_process_memory();
                    imgD_ = None
                    fix_ = None
                    clear_variables((imgD_, fix_))
                except MemoryError as err:
                    print(f'Memory Error on file {folder}/{fish}: {err}')
    return None


def registration(is_largefile=True):
    from pathlib import Path
    from fish_proc.pipeline.preprocess import motion_correction
    from skimage.io import imread, imsave
    dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
    if 'index' in dat_xls_file.columns:
        dat_xls_file = dat_xls_file.drop('index', axis=1)
    dat_xls_file['folder'] = dat_xls_file['folder'].astype(int).apply(str)

    for _, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        save_folder = dat_folder + f'{folder}/{fish}/Data'
        print(f'checking file {folder}/{fish}')
        if not os.path.isfile(save_folder+'/imgDMotion.tif') and os.path.isfile(save_folder + '/motion_fix_.npy'):
            if not os.path.isfile(save_folder+'/proc_registr.tmp'):
                Path(save_folder+'/proc_registr.tmp').touch()
                print(f'process file {folder}/{fish}')
                imgD_ = imread(save_folder+'/imgDNoMotion.tif').astype('float32')
                fix_ = np.load(save_folder + '/motion_fix_.npy').astype('float32')
                if is_largefile:
                    len_D_ = len(imgD_)//2
                    motion_correction(imgD_[:len_D_], fix_, save_folder, ext='0')
                    get_process_memory();
                    motion_correction(imgD_[len_D_:], fix_, save_folder, ext='1')
                    get_process_memory();
                    s_ = [np.load(save_folder+'/imgDMotion%d.npy'%(__)) for __ in range(2)]
                    s_ = np.concatenate(s_, axis=0).astype('float32')
                    imsave(save_folder+'/imgDMotion.tif', s_, compress=1)
                    s_ = None
                    clear_variables(s_)
                    os.remove(save_folder+'/imgDMotion0.npy')
                    os.remove(save_folder+'/imgDMotion1.npy')
                else:
                    motion_correction(imgD_, fix_, save_folder)
                    get_process_memory();
                imgD_ = None
                fix_ = None
                clear_variables((imgD_, fix_))
                Path(save_folder+'/finished_registr.tmp').touch()
    return None


def video_detrend():
    from fish_proc.pipeline.denoise import detrend
    from pathlib import Path
    from multiprocessing import cpu_count
    from skimage.io import imsave, imread

    dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
    for _, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        save_folder = dat_folder + f'{folder}/{fish}/Data'
        print(f'checking file {folder}/{fish}')
        if not os.path.isfile(save_folder+'/Y_d.tif') and not os.path.isfile(save_folder+'/proc_detrend.tmp'):
            if os.path.isfile(save_folder+'/finished_registr.tmp'):
                Path(save_folder+'/proc_detrend.tmp').touch()
                Y = imread(save_folder+'/imgDMotion.tif').astype('float32')
                Y = Y.transpose([1,2,0])

                n_split = min(Y.shape[0]//cpu_count(), 8)
                if n_split <= 1:
                    n_split = 2                
                Y_len = Y.shape[0]//2
                detrend(Y[:Y_len, :, :], save_folder, n_split=n_split//2, ext='0')
                detrend(Y[Y_len:, :, :], save_folder, n_split=n_split//2, ext='1')
                Y = None
                clear_variables(Y)
                get_process_memory();
                
                Path(save_folder+'/finished_detrend.tmp').touch()
    return None

def local_pca():
    from fish_proc.pipeline.denoise import denose_2dsvd
    from pathlib import Path
    from skimage.external.tifffile import imsave, imread

    dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        image_folder = f'/nrs/ahrens/Takashi/0{folder}/{fish}/'
        save_folder = dat_folder + f'{folder}/{fish}/Data'
        print(f'checking file {folder}/{fish}')
        if not os.path.isfile(save_folder+'/proc_local_denoise.tmp'):
            if os.path.isfile(save_folder+'/finished_detrend.tmp'):
                Path(save_folder+'/proc_local_denoise.tmp').touch()

                if os.path.isfile(f'{save_folder}/Y_d.npy'):
                    Y_d = np.load(f'{save_folder}/Y_d.npy').astype('float32')
                elif os.path.isfile(f'{save_folder}/Y_d.tif'):
                    Y_d = imread(f'{save_folder}/Y_d.tif')

                Y_d_ave = Y_d.mean(axis=-1, keepdims=True) # remove mean
                Y_d_std = Y_d.std(axis=-1, keepdims=True) # normalization
                Y_d = (Y_d - Y_d_ave)/Y_d_std
                Y_d = Y_d.astype('float32')
                np.savez_compressed(f'{save_folder}/Y_2dnorm', Y_d_ave=Y_d_ave, Y_d_std=Y_d_std)
                Y_d_ave = None
                Y_d_std = None
                clear_variables((Y_d_ave, Y_d_std))
                get_process_memory();

                for n, Y_d_ in enumerate(np.array_split(Y_d, 10, axis=-1)):
                    denose_2dsvd(Y_d_, save_folder, ext=f'{n}')

                Y_d_ = None
                Y_d = None
                clear_variables(Y_d)
                get_process_memory();
                Path(save_folder+'/finished_local_denoise.tmp').touch()
    return None

if __name__ == '__main__':
    if len(sys.argv)>1:
        eval(sys.argv[1]+"()")
    else:
        local_pca()

    # update_table(update_ods = False)
    # swim()
    # pixel_denoise()
    # registration()
    # video_detrend()
    # local_pca()
