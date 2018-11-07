import numpy as np
import pandas as pd
import os, sys
from trefide.temporal import TrendFilter

dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/'

def single_x(voltr):
    if voltr.ndim>1:
        voltr = voltr.reshape(-1)
    voltr_ = voltr[600:]
    n_spk = np.zeros(len(voltr_)).astype('bool')
    voltr_ = roll_scale(voltr_, win_=50001)
    x_, _ = prepare_sequences_center(voltr_, n_spk, window_length, peak_wid=2)
    return voltr_[np.newaxis, :], np.expand_dims(x_, axis=0)


def voltr2spike(voltrs, window_length, m):
    from fish_proc.utils.np_mp import parallel_to_chunks
    import time
    start = time.time()
    voltr_list, x_list = parallel_to_chunks(single_x, voltrs)
    print(time.time() - start)
    n_, len_ = voltr_list.shape
    spk1_list = np.empty(voltr_list.shape)
    spk2_list = np.empty(voltr_list.shape)
    spk_list = np.empty(voltr_list.shape)
    spkprob_list = np.empty(voltr_list.shape)
    for _, (voltr_, x_) in enumerate(zip(voltr_list, x_list)):
        start = time.time()
        pred_x_test = m.predict(x_)
        spk_, spkprob = detected_window_max_spike(pred_x_test, voltr_, window_length = window_length, peak_wid=2, thres=0.5)
        spk1, spk2 = cluster_spikes(spk_, spkprob, voltr_)
        spk_list[_, :] = spk_
        spk1_list[_, :] = spk1
        spk2_list[_, :] = spk2
        spkprob_list[_, :] = spkprob
        print(f'Spike detection for neuron #{_} is done......')
        print(time.time() - start)
    print('Spike detection done for all neurons')
    return spk_list, spkprob_list, spk1_list, spk2_list, voltr_list


def tf_filter(_):
    from trefide.temporal import TrendFilter
    spk__, voltr_, voltr= _
    filters = TrendFilter(len(voltr_))
    tspk = np.where(spk__>0)[0]
    tspk_win = tspk[:, None] + np.arange(-3, 3)[None, :]
    tspk_win = tspk_win.reshape(-1)
    nospike = np.zeros(spk__.shape)
    nospike[tspk_win] = 1
    tspk_ = np.where(nospike==0)[0]
    int_voltr_ = voltr_.copy()
    int_voltr_[tspk_win] = np.interp(tspk_win, tspk_, voltr_[tspk_])
    denoised_voltr_ = filters.denoise(int_voltr_)

    int_voltr_ = voltr[600:].copy()
    int_voltr_[tspk_win] = np.interp(tspk_win, tspk_, voltr[600:][tspk_])
    denoised_voltr = filters.denoise(int_voltr_)
    out = (denoised_voltr_, denoised_voltr)
    return np.asarray(out[0])[np.newaxis,:], np.asarray(out[1])[np.newaxis,:]


def voltr2subvolt():
    '''
    This one can be benefited from multiple cores.
    '''
    from pathlib import Path
    import multiprocessing as mp

    dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
    if 'index' in dat_xls_file.columns:
        dat_xls_file = dat_xls_file.drop('index', axis=1)
    dat_xls_file['folder'] = dat_xls_file['folder'].astype(int).apply(str)

    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        save_folder = dat_folder + f'{folder}/{fish}/Data'

        if os.path.isfile(save_folder+'/finished_subvolt.tmp'):
            continue
        if not os.path.isfile(save_folder+'/finished_spikes.tmp'):
            print(f'Spike file is not ready for {save_folder}')
            continue
        if not os.path.isfile(save_folder+'/proc_subvolt.tmp'):
            Path(save_folder+'/proc_subvolt.tmp').touch()
            print(f'Processing {save_folder}')
            _ = np.load(f'{save_folder}/Voltr_spikes.npz')
            voltrs = _['voltrs']
            spk = _['spk']
            spkprob = _['spkprob']
            spk1 = _['spk1']
            spk2 = _['spk2']
            voltr_ = _['voltr_']
            n_, len_ = voltrs.shape
            spk__list = spk1+spk2
            dat_ = [(spk__list[_, :], voltr_[_, :], voltrs[_, :]) for _ in range(n_)]
            mp_count = min(mp.cpu_count(), n_)
            pool = mp.Pool(processes=mp_count)
            individual_results = pool.map(tf_filter, dat_)
            pool.close()
            pool.join()
            results = ()
            for i_tuple in range(len(individual_results[0])):
                results = results + (np.concatenate([_[i_tuple] for _ in individual_results]), )

            np.savez_compressed(f'{save_folder}/Voltr_subvolt', norm_subvolt=results[0], subvolt_=results[1])
            Path(save_folder+'/finished_subvolt.tmp').touch()

    return None


def voltron():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from skimage.external.tifffile import imsave, imread
    from fish_proc.utils.demix import recompute_nmf, recompute_C_matrix, pos_sig_correction
    import pickle

    sns.set(font_scale=2)
    sns.set_style("white")
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

        if os.path.isfile(save_folder+'/finished_voltr.tmp'):
            continue

        Path(save_folder+'/proc_voltr.tmp').touch()
        Y_trend_ave = np.load(f'{save_folder}/Y_trend_ave.npy')

        print('update components images')
        with open(f'{save_folder}/period_Y_demix_rlt.pkl', 'rb') as f:
            rlt_ = pickle.load(f)
        d1, d2 = Y_trend_ave.shape
        mask = np.empty((d2, d1))
        mask[:] = False
        pixel = 4
        mask[:pixel, :]=True
        mask[-pixel:,:]=True
        mask[:, :pixel]=True
        mask[:,-pixel:]=True
        mask = mask.astype('bool')
        A = rlt_['fin_rlt']['a'].copy()
        A[mask.reshape(-1),:]=0
        A_ = A[:, (A>0).sum(axis=0)>40] # min pixel = 40
        A_comp = np.zeros(A_.shape[0])
        A_comp[A_.sum(axis=-1)>0] = np.argmax(A_[A_.sum(axis=-1)>0, :], axis=-1) + 1
        plt.figure(figsize=(8,4))
        plt.imshow(Y_trend_ave, cmap=plt.cm.gray)
        plt.imshow(A_comp.reshape(d2, d1).T, cmap=plt.cm.nipy_spectral_r, alpha=0.7)
        for n, nA in enumerate(A_.T):
            nA = nA.reshape(d2, d1).T
            pos = np.where(nA>0);
            pos0 = pos[0];
            pos1 = pos[1];
            plt.text(pos1.mean(), pos0.mean(), f"{n}", fontsize=15)
        plt.title('Components')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_image_folder}/Demixed_components.png')

        plt.figure(figsize=(8,4))
        plt.imshow(A_.sum(axis=-1).reshape(d2, d1).T)
        for n, nA in enumerate(A_.T):
            nA = nA.reshape(d2, d1).T
            pos = np.where(nA>0);
            pos0 = pos[0];
            pos1 = pos[1];
            plt.text(pos1.mean(), pos0.mean(), f"{n}", fontsize=15, color='w')
        plt.title('Components weights')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_image_folder}/Demixed_components_weights.png')


        print('Start computing voltron data')
        _ = np.load(f'{save_folder}/Y_2dnorm.npz')
        Y_d_std= _['Y_d_std']
        # if os.path.getsize(f'{save_folder}/Y_svd.tif')/1024/1024/1024>=50.:
        #     continue
        Y_svd = imread(f'{save_folder}/Y_svd.tif').astype('float32')
        mov = -Y_svd*Y_d_std
        b = rlt_['fin_rlt']['b']
        fb = rlt_['fin_rlt']['fb']
        ff = rlt_['fin_rlt']['ff']
        dims = mov.shape
        if fb is not None:
            b_ = np.matmul(fb, ff.T)+b
        else:
            b_ = b
        mov = pos_sig_correction(mov, -1)
        mov = mov - b_.reshape((dims[0], dims[1], len(b_)//dims[0]//dims[1]), order='F')
        C_ = recompute_C_matrix(mov, A_)
        base_ = recompute_C_matrix(Y_trend_ave[:, :, np.newaxis], A_)
        np.savez_compressed(f'{save_folder}/Voltr_raw', A_=A_, C_=C_, base_=base_)
        Path(save_folder+'/finished_voltr.tmp').touch()
    return None


def main():
    '''
    There seems to be a limitation of cores keras can use, 4 - 8 cores are enough for this one.
    '''
    import keras
    from keras.models import load_model
    from fish_proc.spikeDetectionNN.spikeDetector import prepare_sequences_center
    from fish_proc.spikeDetectionNN.utils import detected_window_max_spike
    from fish_proc.spikeDetectionNN.utils import roll_scale
    from fish_proc.spikeDetectionNN.utils import cluster_spikes
    from glob import glob
    from pathlib import Path

    trained_model = '/groups/ahrens/home/weiz/fish_processing/notebooks/simEphysImagingData/partly_trained_spikeDetector_2018_09_27_01_25_36.h5'
    m = load_model(trained_model)
    window_length = 41
    print(keras.__version__)

    dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
    if 'index' in dat_xls_file.columns:
        dat_xls_file = dat_xls_file.drop('index', axis=1)
    dat_xls_file['folder'] = dat_xls_file['folder'].astype(int).apply(str)

    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        save_folder = dat_folder + f'{folder}/{fish}/Data'

        if os.path.isfile(save_folder+'/finished_spikes.tmp'):
            continue
        if not os.path.isfile(save_folder+'/proc_spikes.tmp'):
            Path(save_folder+'/proc_spikes.tmp').touch()
            _ = np.load(f'{save_folder}/Voltr_raw.npz')
            A_ = _['A_']
            C_ = _['C_']
            base_ = _['base_']
            voltrs = C_/(C_.mean(axis=-1, keepdims=True)+base_)
            spk_list, spkprob_list, spk1_list, spk2_list, voltr_list = voltr2spike(voltrs, window_length, m)
            np.savez_compressed(f'{save_folder}/Voltr_spikes', voltrs=voltrs, \
                                spk=spk_list, spkprob=spkprob_list, spk1=spk1_list, \
                                spk2=spk2_list, voltr_=voltr_list)
            Path(save_folder+'/finished_spikes.tmp').touch()
    return None


if __name__ == "__main__":
    if len(sys.argv)>1:
        eval(sys.argv[1]+"()")
    else:
        # main()
        voltr2subvolt()
