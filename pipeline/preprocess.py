"""
@author: modified by Ziqiang Wei, 08/30/2018
"""
import numpy as np
import h5py
from skimage import io
# import os, sys
# fish_path = os.path.abspath(os.path.join('..'))
# if fish_path not in sys.path:
#     sys.path.append(fish_path)
import matplotlib.pyplot as plt

def pixel_denoise(folderName, imgFileName, fishName, cameraNoiseMat, plot_en=False):
    from ..utils import getCameraInfo
    from ..pixelwiseDenoising.simpleDenioseTool import simpleDN
    from scipy.ndimage.filters import median_filter
    from ..utils.memory import get_process_memory, clear_variables

    cameraInfo = getCameraInfo.getCameraInfo(folderName)
    pixel_x0, pixel_x1, pixel_y0, pixel_y1 = [int(_) for _ in cameraInfo['camera_roi'].split('_')]
    pixel_x = (pixel_x0, pixel_x1)
    pixel_y = (pixel_y0, pixel_y1)
    imgFile = os.path.join(folderName, imgFileName)
    imgStack = io.imread(imgFile).astype('float32')
    if plot_en:
        plt.figure(figsize=(4, 3))
        plt.imshow(imgStack[0], cmap='gray')
        plt.savefig(fishName + '/Raw_frame_0.png')
    offset = np.load(cameraNoiseMat +'/offset_mat.npy').astype('float32')
    gain = np.load(cameraNoiseMat +'/gain_mat.npy').astype('float32')
    offset_ = offset[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    gain_ = gain[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]

    ## simple denoise
    imgD = simpleDN(imgStack, offset=offset_, gain=gain_)

    ## memory release ---
    imgStack = None
    clear_variables(imgStack)
    ## smooth dead pixels
    win_ = 3
    imgD_ = median_filter(imgD, size=(1, win_, win_))
    np.save(fishName+'/imgDNoMotion', imgD_)
    return imgD_

def pixel_denoise_img_seq(folderName, fishName, cameraNoiseMat, plot_en=False):
    from ..utils import getCameraInfo
    from ..pixelwiseDenoising.simpleDenioseTool import simpleDN
    from scipy.ndimage.filters import median_filter
    from glob import glob

    cameraInfo = getCameraInfo.getCameraInfo(folderName)
    pixel_x0, pixel_x1, pixel_y0, pixel_y1 = [int(_) for _ in cameraInfo['camera_roi'].split('_')]
    pixel_x = (pixel_x0, pixel_x1)
    pixel_y = (pixel_y0, pixel_y1)
    imgFiles = sorted(glob(folderName+'TM*_CM*_CHN*.tif'))
    imgStack = np.concatenate([io.imread(_).astype('float32') for _ in imgFiles], axis=0).astype('float32')
    if plot_en:
        plt.figure(figsize=(4, 3))
        plt.imshow(imgStack[0], cmap='gray')
        plt.savefig(fishName + '/Raw_frame_0.png')
    offset = np.load(cameraNoiseMat +'/offset_mat.npy').astype('float32')
    gain = np.load(cameraNoiseMat +'/gain_mat.npy').astype('float32')
    offset_ = offset[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    gain_ = gain[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]

    ## simple denoise
    imgD = simpleDN(imgStack, offset=offset_, gain=gain_)
    ## smooth dead pixels
    win_ = 3
    imgD_ = median_filter(imgD, size=(1, win_, win_))
    np.save(fishName+'/imgDNoMotion', imgD_)
    return imgD_

def regidStacks(move, fix=None, trans=None):
    if move.ndim < 3:
        move = move[np.newaxis, :]
    trans_move = move.copy()
    move_list = []
    for nframe, move_ in enumerate(move):
        trans_affine = trans.estimate_rigid2d(fix, move_)
        trans_mat = trans_affine.affine
        trans_move[nframe] = trans_affine.transform(move_)
        move_list.append([trans_mat[0, 1]/trans_mat[0, 0], trans_mat[0, 2], trans_mat[1, 2]])
    return trans_move, move_list

def motion_correction(imgD_, fix_, fishName):
    from ..imageRegistration.imTrans import ImAffine
    from ..utils.np_mp import parallel_to_chunks
    from ..utils.memory import get_process_memory, clear_variables

    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.ss_sigma_factor = 1.0

    print('memory usage before processing -- ')
    get_process_memory();

    imgDMotion, imgDMotionVar = parallel_to_chunks(regidStacks, imgD_, fix=fix_, trans=trans)
    # imgStackMotion, imgStackMotionVar = parallel_to_chunks(regidStacks, imgStack, fix=fix, trans=trans)
    # np.save('tmpData/imgStackMotion', imgStackMotion)
    # np.save('tmpData/imgStackMotionVar', imgStackMotionVar)
    np.save(fishName+'/imgDMotion', imgDMotion)
    np.save(fishName+'/imgDMotionVar', imgDMotionVar)

    print('memory usage after processing -- ')
    get_process_memory();
    print('relase memory')
    imgDMotion = None
    imgDMotionVar = None
    clear_variables((imgDMotion, imgDMotionVar))

    return None

def compute_dff(sig):
    bg = np.percentile(sig, 5, axis=1)
    return (sig.mean(axis=1)-sig.mean(axis=1).mean())/(sig.mean(axis=1).mean()-bg)

if __name__ == '__main__':
    folderName = '/groups/ahrens/ahrenslab/Takashi/toZiqiang/02212018Fish2-1/'
    imgFileName = 'Raw_stack.tif'
    fishName = '02212018Fish2-1_Raw_stack'
    cameraNoiseMat = '/groups/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'
    if not os.path.exists(fishName):
        os.mkdir(fishName)
    imgD_ = pixel_denoise(folderName, imgFileName, fishName, cameraNoiseMat, plot_en=False)
    t_ = len(imgD_)//2
    win_ = 150
    fix_ = imgD_[t_-win_:t_+win_].mean(axis=0)
    savefix_ = True
    if savefix_:
        np.save(fishName + '/motion_fix_', fix_)
    motion_correction(imgD_, fix_, fishName)
