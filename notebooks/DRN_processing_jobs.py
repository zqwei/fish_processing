import numpy as np
import pandas as pd
import os, sys
from fish_proc.utils.memory import get_process_memory, clear_variables

dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/'
cameraNoiseMat = '/groups/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'


# def update_table(update_ods = False):
#     '''
#     Update Voltron Log_DRN_Exp.csv
#     run when new data is added
#     '''
#     if update_ods:
#         dat_xls_file = pd.read_excel(dat_folder+'Voltron Log_DRN_Exp.xlsx')
#         dat_xls_file = dat_xls_file.dropna(how='all').reset_index()
#         dat_xls_file['folder'] = dat_xls_file['folder'].astype('int').astype('str')
#         dat_xls_file['finished'] = False
#         dat_xls_file.to_csv('../Voltron Log_DRN_Exp.csv')
#     return None

def monitor_process():
    '''
    Update Voltron Log_DRN_Exp.csv
    monitor process of processing
    '''
    dat_xls_file = pd.read_csv('../Voltron Log_DRN_Exp.csv', index_col=0)
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
    dat_xls_file.to_csv('../Voltron Log_DRN_Exp.csv')
    return None


def swim():
    '''
    Processing swim using TK's code
    '''
    from fish_proc.utils.ep import process_swim
    dat_xls_file = pd.read_csv('../Voltron Log_DRN_Exp.csv', index_col=0)
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
    from fish_proc.pipeline.preprocess import pixel_denoise, pixel_denoise_img_seq
    dat_xls_file = pd.read_csv('../Voltron Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        image_folder = row['rootDir'] + f'{folder}/{fish}/'
        fish_folder = dat_folder + f'{folder}/{fish}/'
        save_folder = dat_folder + f'{folder}/{fish}/Data'

        if os.path.exists(image_folder):
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
    '''
    Generate imgDMotion.tif
    '''
    from pathlib import Path
    from fish_proc.pipeline.preprocess import motion_correction
    from skimage.io import imread, imsave
    dat_xls_file = pd.read_csv('../Voltron Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

    for index, row in dat_xls_file.iterrows():
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
                    imgD_ = None
                    fix_ = None
                    clear_variables((imgD_, fix_))
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
                    s_ = np.load(save_folder+'/imgDMotion.npy').astype('float32')
                    imsave(save_folder+'/imgDMotion.tif', s_, compress=1)
                    s_ = None
                    clear_variables(s_)
                    os.remove(save_folder+'/imgDMotion.npy')
                Path(save_folder+'/finished_registr.tmp').touch()
    return None


def video_detrend():
    from fish_proc.pipeline.denoise import detrend
    from pathlib import Path
    from multiprocessing import cpu_count
    from skimage.io import imsave, imread
    dat_xls_file = pd.read_csv('../Voltron Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        save_folder = dat_folder + f'{folder}/{fish}/Data'
        print(f'checking file {folder}/{fish}')
        if os.path.isfile(save_folder+'/finished_detrend.tmp'):
            continue

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
                Y = []
                Y.append(np.load(save_folder+'/Y_d0.npy').astype('float32'))
                Y.append(np.load(save_folder+'/Y_d1.npy').astype('float32'))
                Y = np.concatenate(Y, axis=0).astype('float32')
                imsave(save_folder+'/Y_d.tif', Y, compress=1)
                Y = None
                clear_variables(Y)
                get_process_memory();
                os.remove(save_folder+'/Y_d0.npy')
                os.remove(save_folder+'/Y_d1.npy')
                # os.remove(save_folder+'/Y_trend0.npy')
                # os.remove(save_folder+'/Y_trend1.npy')
                Path(save_folder+'/finished_detrend.tmp').touch()
    return None


def local_pca():
    from fish_proc.pipeline.denoise import denose_2dsvd
    from pathlib import Path
    from skimage.external.tifffile import imsave, imread
    dat_xls_file = pd.read_csv('../Voltron Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        image_folder = f'/nrs/ahrens/Takashi/0{folder}/{fish}/'
        save_folder = dat_folder + f'{folder}/{fish}/Data'
        print(f'checking file {folder}/{fish}')
        if os.path.isfile(save_folder+'/finished_local_denoise.tmp'):
            continue

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
    dat_xls_file = pd.read_csv('../Voltron Log_DRN_Exp.csv', index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        image_folder = f'/nrs/ahrens/Takashi/0{folder}/{fish}/'
        save_folder = dat_folder + f'{folder}/{fish}/Data'
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

            mov_ = Y_svd_*Y_d_std + np.random.normal(size=Y_svd_.shape)*0.7
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
    dat_xls_file = pd.read_csv('../Voltron Log_DRN_Exp.csv', index_col=0)
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
    save_folder = dat_folder + f'{folder}/{fish}/Data'
    save_image_folder = dat_folder + f'{folder}/{fish}/Results'
    if not os.path.exists(save_image_folder):
        os.makedirs(save_image_folder)
    print('=====================================')
    print(save_folder)

    if os.path.isfile(save_folder+f'/finished_demix{ext}.tmp'):
        return None

    if not os.path.isfile(save_folder+f'/proc_demix{ext}.tmp'):
        Path(save_folder+f'/proc_demix{ext}.tmp').touch()
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
        get_process_memory();

        # get sparse data
        Y_d_std_ = Y_d_std[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max(), :]
        mov_ = mov_*Y_d_std_ + np.random.normal(size=mov_.shape)*0.7
        mov_ = -mov_.astype('float32')
        mask_ = mask[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max()]
        mov_[mask_]=0
        get_process_memory();
        d1, d2, _ = mov_.shape

        # get local correlation distribution
        Cn, _ = correlation_pnr(mov_, skip_pnr=True)
        get_process_memory();

        pass_num = 4
        cut_off_point=np.percentile(Cn.ravel(), [99, 95, 85, 65])
        rlt_= sup.demix_whole_data(mov_, cut_off_point, length_cut=[10,15,15,15],
                                   th=[1,1,1,1], pass_num=pass_num, residual_cut = [0.6,0.6,0.6,0.6],
                                   corr_th_fix=0.3, max_allow_neuron_size=0.05, merge_corr_thr=cut_off_point[-1],
                                   merge_overlap_thr=0.6, num_plane=1, patch_size=[40, 40], plot_en=False,
                                   TF=False, fudge_factor=1, text=False, bg=False, max_iter=60,
                                   max_iter_fin=100, update_after=20)

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
        Path(save_folder+f'/finished_demix{ext}.tmp').touch()
    return None


if __name__ == '__main__':
    if len(sys.argv)>1:
        ext = ''
        if len(sys.argv)>2:
            ext = sys.argv[2]
        eval(sys.argv[1]+f"({ext})")
    else:
        monitor_process()
