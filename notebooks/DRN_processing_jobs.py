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
    print(dat_xls_file[dat_xls_file['registration']==False])
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


def registration(is_largefile=False):
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
                if is_largefile:
                    len_D_ = len(imgD_)//2
                    motion_correction(imgD_[:len_D_], fix_, save_folder, ext='0')
                    get_process_memory();
                    motion_correction(imgD_[len_D_:], fix_, save_folder, ext='1')
                    get_process_memory();
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
    
    dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        image_folder = f'/nrs/ahrens/Takashi/0{folder}/{fish}/'
        save_folder = dat_folder + f'{folder}/{fish}/Data'
        print(f'checking file {folder}/{fish}')
        if not os.path.isfile(save_folder+'/Y_d.npy') and not os.path.isfile(save_folder+'/proc_detrend.tmp'):
            if os.path.isfile(save_folder+'/imgDMotionVar.npy') or os.path.isfile(save_folder+'/Data/finished_registr.tmp'):
                Path(save_folder+'/proc_detrend.tmp').touch()
                if not os.path.isfile(save_folder+'/imgDMotionVar.npy'):
                    Y1 = np.load(save_folder+'/imgDMotion0.npy').astype('float32')
                    Y2 = np.load(save_folder+'/imgDMotion1.npy').astype('float32')
                    Y = np.concatenate((Y1, Y2), axis=0)
                    Y1 = None
                    Y2 = None
                    clear_variables((Y1, Y2))
                else:
                    Y = np.load(save_folder+'/imgDMotion.npy').astype('float32')
                Y = Y.transpose([1,2,0])
                detrend(Y, save_folder, n_split=4)
                Y = None
                clear_variables(Y)
                get_process_memory();
                Path(save_folder+'/finished_detrend.tmp').touch()
    return None

if __name__ == '__main__':
    if len(sys.argv)>1:
        eval(sys.argv[1]+"()")
    else:
        video_detrend()
    
    # update_table(update_ods = False)
    # swim()
    # pixel_denoise()
    # registration()
    # video_detrend()
