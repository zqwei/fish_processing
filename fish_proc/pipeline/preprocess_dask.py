
"""
-- dask
@author: modified by Salma  Elmalaki, 12/5/2018
"""
import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
import dask
from skimage import io

def load_img_seq_dask(image_folder):
    from glob import glob

    imread = dask.delayed(io.imread, pure=True)  # Lazy version of imread
    imgFiles = sorted(glob(image_folder + 'TM*_CM*_CHN*.tif'))
    #start_time = time.time()
    lazy_images = [imread(img).astype('float32') for img in imgFiles]  # Lazily evaluate imread on each path

    sample = lazy_images[0].compute()  # load the first image (assume rest are same shape/dtype)
    arrays = [da.from_delayed(lazy_image,  # Construct a small Dask array
                              dtype=sample.dtype,  # for every lazy value
                              shape=sample.shape)
              for lazy_image in lazy_images]

    imgStack = da.concatenate(arrays, axis=0).astype('float32')
    #imgStackDask = imgStack.compute() #- compute in the last step
    #print("--- %s seconds for image stack creation: dask ---" % (time.time() - start_time))
    return imgStack



def pixel_denoise(folderName, imgFileName, fishName, cameraNoiseMat, plot_en=False):
    from ..utils import getCameraInfo
    from ..pixelwiseDenoising.simpleDenioseTool import simpleDN
    from scipy.ndimage.filters import median_filter
    from ..utils.memory import clear_variables
    import os

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
    # np.save(fishName+'/imgDNoMotion', imgD_)
    io.imsave(fishName+'/imgDNoMotion.tif', imgD_, compress=1)
    return imgD_


#@profile  # --salma for memory profiling
def pixel_denoise_img_seq(folderName, fishName, cameraNoiseMat, plot_en=False):
    from ..utils import getCameraInfo
    from ..pixelwiseDenoising.simpleDenioseTool_dask import simpleDN
    import dask_ndfilters as daf
    import time  # --salma

    cameraInfo = getCameraInfo.getCameraInfo(folderName)
    pixel_x0, pixel_x1, pixel_y0, pixel_y1 = [int(_) for _ in cameraInfo['camera_roi'].split('_')]
    pixel_x = (pixel_x0, pixel_x1)
    pixel_y = (pixel_y0, pixel_y1)

    start_time = time.time()  # --salma
    imgStack = load_img_seq_dask(folderName)
    print("--- %s seconds for image stack creation ---" % (time.time() - start_time))  # --salma

    if plot_en:
        plt.figure(figsize=(4, 3))
        plt.imshow(imgStack[0], cmap='gray')
        plt.savefig(fishName + '/Raw_frame_0.png')

    #start_time = time.time()  # --salma
    offset = np.load(cameraNoiseMat +'/offset_mat.npy').astype('float32')
    #print("--- %s seconds for loading offset data ---" % (time.time() - start_time))  # --salma
    #start_time = time.time()  # --salma
    gain = np.load(cameraNoiseMat +'/gain_mat.npy').astype('float32')
    #print("--- %s seconds for loading gain data ---" % (time.time() - start_time))  # --salma
    offset_ = offset[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    gain_ = gain[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]

    ## simple denoise
    start_time = time.time()  # --salma
    imgD = simpleDN(imgStack, offset=offset_, gain=gain_)
    print("--- %s seconds for simple DN function ---" % (time.time() - start_time))  # --salma
    ### smooth dead pixels

    win_ = 3
    start_time = time.time()  # --salma
    imgDFiltered = daf.median_filter(imgD,  size=(1, win_, win_))
    imgDFiltered = imgDFiltered.compute() # - compute here before saving
    print("--- %s seconds for median filter ---" % (time.time() - start_time))  # --salma

    # np.save(fishName+'/imgDNoMotion', imgD_)
    start_time = time.time()  # --salma
    io.imsave(fishName+'/imgDNoMotionDask.tif', imgDFiltered, compress=1)
    print("--- %s seconds for saving imgDNoMotionDask file ---" % (time.time() - start_time))  # --salma


    if plot_en:
        plt.figure(figsize=(4, 3))
        plt.imshow(imgDFiltered[0], cmap='gray')
        plt.savefig(fishName + '/Denoised_frame_0.png')


    return imgDFiltered

######### Registration bag helpers ##########################
def rigid_stacks_element(move_frame, fix=None, trans=None):

    #print("rigid stacks element: "+str(move_frame.shape))
    trans_affine = trans.estimate_rigid2d(fix, move_frame)
    trans_mat = trans_affine.affine
    trans_move = trans_affine.transform(move_frame)
    move_var = [trans_mat[0, 1]/trans_mat[0, 0], trans_mat[0, 2], trans_mat[1, 2]]

    return trans_move, move_var


def motion_correction_element(image_frame, fix):
    from fish_proc.imageRegistration.imTrans import ImAffine

    #print("Image_frame shape: "+str(image_frame.shape))
    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.ss_sigma_factor = 1.0

    imgDMotion_element, imgDMotionVar_element = rigid_stacks_element(image_frame, fix=fix, trans=trans)

    return imgDMotion_element, imgDMotionVar_element
###############################################################


######### Registration manual batch helpers ###################
def batch_motion_correction(seq, fix):
    sub_results = []
    for image_frame in seq:
        sub_results.append(motion_correction_dask_array(image_frame, fix))
    return sub_results

def rigid_stacks_dask_array(move, fix=None, trans=None):
    # start_time = time.time()
    if move.ndim < 3:
        move = move[np.newaxis, :]

    trans_move = move.copy()
    move_list = []

    for nframe, move_ in enumerate(move):
        trans_affine = trans.estimate_rigid2d(fix, move_)
        trans_mat = trans_affine.affine
        trans_move[nframe] = trans_affine.transform(move_)
        move_list.append([trans_mat[0, 1] / trans_mat[0, 0], trans_mat[0, 2], trans_mat[1, 2]])

    # print("--- %s seconds for rigidstacks---" % (time.time() - start_time))  # --salma
    return trans_move, move_list


def motion_correction_dask_array(image_frame, fix):
    from fish_proc.imageRegistration.imTrans import ImAffine

    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.ss_sigma_factor = 1.0

    imgDMotion_element, imgDMotionVar_element = rigid_stacks_dask_array(image_frame, fix=fix, trans=trans)

    return imgDMotion_element, imgDMotionVar_element
###############################################################

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
