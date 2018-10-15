import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
sns.set_style("white")
from fish_proc.utils.memory import get_process_memory, clear_variables
get_process_memory();
import pandas as pd

from pathlib import Path
from skimage.external.tifffile import imsave, imread
from funimag import superpixel_analysis as sup
from fish_proc.utils.demix import recompute_nmf
from trefide.extras.util_plot import correlation_pnr
import caiman as cm
import pickle

dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/'
dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)


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
