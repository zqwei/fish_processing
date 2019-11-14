from .preprocess import pixel_denoise, motion_correction
from .denoise import denose_2dsvd, detrend
from ..utils.memory import get_process_memory, clear_variables
from ..demix.superpixel_analysis import demix_whole_data
import os, pickle
import numpy as np

class pipe:
    def __init__(self):
        folderName = '/groups/ahrens/ahrenslab/Takashi/toZiqiang/02212018Fish2-1/'
        imgFileName = 'Raw_stack.tif'
        fishName = '02212018Fish2-1_Raw_stack'
        cameraNoiseMat = '/groups/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'
        if not os.path.exists(fishName):
            os.mkdir(fishName)
        savefix_ = True
        plot_en = True
        registration_fix = None
        detrend_split = 32
        denoise_patch = [10, 10]
        stim_knots=None
        stim_delta=0
        mov_sign = -1


    def preprocess(self):
        # pixel denoise
        imgD_ = pixel_denoise(self.olderName, self.imgFileName, self.fishName,
                              self.cameraNoiseMat, plot_en=self.plot_en)

        # motion correction
        if self.registration_fix is None:
            t_ = len(imgD_)//2
            win_ = min(150, t_//4)
            fix_ = imgD_[t_-win_:t_+win_].mean(axis=0)
        else:
            fix_ = self.registration_fix
        savefix_ = True
        if savefix_:
            np.save(f'{self.fishName}/motion_fix_', fix_)
        motion_correction(imgD_, fix_, self.fishName)
        return None

    def denoise(self):
        # detrend
        imgDMotion = np.load(f'{self.fishName}/imgDMotion')
        Y = np.asarray(imgDMotion)
        Y = Y.transpose([1,2,0])
        try:
            detrend(Y, self.fishName, n_split = self.detrend_split)
        except ValueError:
            print('Please set self.detrend_split divisible to image height')
        Y = None
        imgDMotion = None
        clear_variables((Y, imgDMotion))
        get_process_memory();

        # denoise
        Y_d = np.load(f'{self.fishName}/Y_d.npy')
        denose_2dsvd(Y_d, self.fishName,
                     nblocks=self.denoise_patch,
                     stim_knots=self.stim_knots,
                     stim_delta=self.stim_delta)
        return None

    def demix(self):
        _ = np.load(f'{self.fishName}/Y_2dsvd.npz')
        Y_amp= _['Y_d_std']
        Y_svd= _['Y_svd']
        _=None
        clear_variables(_)
        get_process_memory();
        mov = Y_svd * Y_amp
        mov_ = mov + np.random.normal(size=Y_svd.shape)*0.7
        pass_num = 3
        rlt_= demix_whole_data(mov_* self.mov_sign,
                               cut_off_point=[0.6,0.6,0.5,0.4],
                               length_cut=[25,25,25,25],
                               th=[2,2,1,1],
                               pass_num=pass_num,
                               residual_cut = [0.6,0.6,0.6, 0.6],
                               corr_th_fix=0.3,
                               max_allow_neuron_size=0.2,
                               merge_corr_thr=0.5,
                               merge_overlap_thr=0.8,
                               num_plane=1,
                               patch_size=[50, 50],
                               plot_en=False,
                               TF=False,
                               fudge_factor=1,
                               text=False,
                               bg=False,
                               max_iter=60,
                               max_iter_fin=100,
                               update_after=4)
        with open(f'{self.fishName}/Y_demix_rlt.pkl', 'wb') as file_:
            pickle.dump(rlt_, file_)
        return None

    def compute_dff(self):
        return None

    def compute_spike(self):
        return None

    def compute_subthreshold(self):
        return None
