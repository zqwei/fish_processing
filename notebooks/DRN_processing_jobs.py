import numpy as np
import pandas as pd
import os
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
    # from subprocess import Popen, PIPE
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
        if os.path.exists(save_folder + '/Data/motion_fix_.npy'):
            dat_xls_file.at[index, 'pixeldenoise'] = True
        if os.path.exists(save_folder+'/Data/imgDMotionVar.npy'):
            dat_xls_file.at[index, 'registration'] = True
    print(dat_xls_file.sum(numeric_only=True))
    dat_xls_file.to_csv(dat_folder + 'Voltron Log_DRN_Exp.csv')
    return None

def swim():
    from fish_proc.utils.ep import process_swim
    dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        image_folder = f'/nrs/ahrens/Takashi/0{folder}/{fish}/'
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
    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        image_folder = f'/nrs/ahrens/Takashi/0{folder}/{fish}/'
        fish_folder = dat_folder + f'{folder}/{fish}/'
        save_folder = dat_folder + f'{folder}/{fish}/Data'
        if os.path.exists(fish_folder):
            print(f'checking file {folder}/{fish}')
            if not os.path.exists(save_folder+'/'):
                os.makedirs(save_folder)
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


def registration():
    from pathlib import Path
    from fish_proc.pipeline.preprocess import motion_correction
    dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
    if 'index' in dat_xls_file.columns:
        dat_xls_file = dat_xls_file.drop('index', axis=1)
    dat_xls_file['folder'] = dat_xls_file['folder'].astype(int).apply(str)

    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        image_folder = f'/nrs/ahrens/Takashi/0{folder}/{fish}/'
        save_folder = dat_folder + f'{folder}/{fish}/Data'
        print(f'checking file {folder}/{fish}')
        if not os.path.isfile(save_folder+'/imgDMotionVar.npy') and os.path.isfile(save_folder + '/motion_fix_.npy'):
            if not os.path.isfile(save_folder+'/proc_registr.tmp'):
                Path(save_folder+'/proc_registr.tmp').touch()
                print(f'process file {folder}/{fish}')
                imgD_ = np.load(save_folder+'/imgDNoMotion.npy').astype('float32')
                fix_ = np.load(save_folder + '/motion_fix_.npy').astype('float32')
                motion_correction(imgD_, fix_, save_folder)
                get_process_memory();
                imgD_ = None
                fix_ = None
                clear_variables((imgD_, fix_))
                Path(save_folder+'/finished_registr.tmp').touch()
    return None

if __name__ == '__main__':
    registration()
