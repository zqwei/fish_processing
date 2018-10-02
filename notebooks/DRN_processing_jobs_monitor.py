import numpy as np
import os, sys
import pandas as pd
import time

def check_status(dat_xls_file):
    from subprocess import Popen, PIPE
    dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/'
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

if __name__ == '__main__':
    dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/'

    dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
    if 'index' in dat_xls_file.columns:
        dat_xls_file = dat_xls_file.drop('index', axis=1)
    dat_xls_file['folder'] = dat_xls_file['folder'].astype(int).apply(str)

    checked = 0

    while_ = False

    if while_:

        while dat_xls_file['registration'].sum()<dat_xls_file.shape[0]:
            check_status(dat_xls_file)
            checked +=1
            print(checked)
            time.sleep(30 * 60)
    else:
        check_status(dat_xls_file)
