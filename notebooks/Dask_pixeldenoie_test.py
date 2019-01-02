"""
Using dask

@author: modified by Salma Elmalaki, Dec 2018

"""


import numpy as np
import pandas as pd
import os, sys
from glob import glob
from skimage import io
import dask.array as da
import dask.array.image as dai
import dask_ndfilters as daf
import dask
import time
from os.path import exists


dat_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/ProcessedData/'
cameraNoiseMat = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/gainMat/gainMat20180208'
root_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/10182018/Fish3-2/'
save_folder = dat_folder + '10182018/Fish3-2/DataSample/using_100_samples/'
sample_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/10182018/Fish3-2/test_sample/'  # Currently 100 samples

#@profile
def load_img_seq_dask():
    imread = dask.delayed(io.imread, pure=True)  # Lazy version of imread

    imgFiles = sorted(glob(sample_folder + 'TM*_CM*_CHN*.tif'))
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

#@profile
def load_img_seq():
    imgFiles = sorted(glob(sample_folder + 'TM*_CM*_CHN*.tif'))
    #start_time = time.time()
    imgStack = np.concatenate([io.imread(_).astype('float32') for _ in imgFiles], axis=0).astype('float32')
    #print("--- %s seconds for image stack creation: np ---" % (time.time() - start_time))
    return imgStack


def simpleDN(img, folder_name='../pixelwiseDenoising/gainMat20180208', pixel_x=None, pixel_y=None, offset=None, gain=None):
    # crop gain and offset matrix to the img size
    assert img.ndim<5 and img.ndim>1, 'Image should be of 2D or 3D or 4D'
    # img should be in form of (t), (z), x, y
    dim_offset = img.ndim - 2
    if pixel_x is None:
        pixel_x = (0, img.shape[dim_offset])
    if pixel_y is None:
        pixel_y = (0, img.shape[dim_offset+1])

    # load gain and offset matrix
    if offset is None:
        assert exists(folder_name)
        offset = np.load(folder_name +'/offset_mat.npy')
    if gain is None:
        gain = np.load(folder_name +'/gain_mat.npy')
    offset_ = offset[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    gain_ = gain[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    for _ in range(dim_offset):
        offset_ = offset_[np.newaxis, :]
        gain_ = gain_[np.newaxis, :]

    print("img shape before computing: "+str(img.shape))
    # compute processesed image
    imgD = (img - offset_) / (gain_ + 1e-12)
    #use dask
    imgD[da.broadcast_to(gain_ < 0.5, imgD.shape)] = 1e-6
    imgD[da.broadcast_to(gain_ > 5, imgD.shape)] = 1e-6
    #imgD[np.broadcast_to(gain_ < 0.5, imgD.shape)] = 1e-6
    #imgD[np.broadcast_to(gain_ > 5, imgD.shape)] = 1e-6
    imgD[imgD <= 0] = 1e-6
    return imgD




def simpleDNTensor(img, folder_name='../pixelwiseDenoising/gainMat20180208', pixel_x=None, pixel_y=None, offset=None, gain=None):
    import tensorflow as tf

    # crop gain and offset matrix to the img size
    assert img.ndim<5 and img.ndim>1, 'Image should be of 2D or 3D or 4D'
    # img should be in form of (t), (z), x, y
    dim_offset = img.ndim - 2
    if pixel_x is None:
        pixel_x = (0, img.shape[dim_offset])
    if pixel_y is None:
        pixel_y = (0, img.shape[dim_offset+1])

    # load gain and offset matrix
    if offset is None:
        assert exists(folder_name)
        offset = np.load(folder_name +'/offset_mat.npy')
    if gain is None:
        gain = np.load(folder_name +'/gain_mat.npy')
    offset_ = offset[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    gain_ = gain[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    for _ in range(dim_offset):
        offset_ = offset_[np.newaxis, :]
        gain_ = gain_[np.newaxis, :]

    print("img shape before computing: "+str(img.shape))
    # compute processesed image
    #imgD = (img - offset_) / (gain_ + 1e-12)

    imgD_t = tf.convert_to_tensor(img)
    offset_t = tf.convert_to_tensor(offset_)
    gain_t = tf.convert_to_tensor(gain_)

    #with tf.device('/gpu:0'):
    Offsetsubtract = tf.subtract(imgD_t, offset_t, name='offset_subtract')
    gainFix = tf.add(gain_t, 1e-12, name="gainfix")
    imgD_t = tf.divide(Offsetsubtract, gainFix, name="divide_by_gain")

    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        imgD = sess.run(imgD_t)

    imgD[np.broadcast_to(gain_ < 0.5, imgD.shape)] = 1e-6
    imgD[np.broadcast_to(gain_ > 5, imgD.shape)] = 1e-6
    imgD[imgD <= 0] = 1e-6
    return imgD


def pixel_denoise_img_seq():
    from fish_proc.utils import getCameraInfo
    #from fish_proc.pixelwiseDenoising.simpleDenioseTool import simpleDN
    from scipy.ndimage.filters import median_filter
    from glob import glob
    from skimage import io
    import time  # --salma

    cameraInfo = getCameraInfo.getCameraInfo(root_folder)
    pixel_x0, pixel_x1, pixel_y0, pixel_y1 = [int(_) for _ in cameraInfo['camera_roi'].split('_')]
    pixel_x = (pixel_x0, pixel_x1)
    pixel_y = (pixel_y0, pixel_y1)

    imgStack = load_img_seq_dask()
    #imgStack = load_img_seq()

    offset = np.load(cameraNoiseMat +'/offset_mat.npy').astype('float32')
    gain = np.load(cameraNoiseMat +'/gain_mat.npy').astype('float32')

    offset_ = offset[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    gain_ = gain[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]


    ## simple denoise
    start_time = time.time()
    imgD = simpleDN(imgStack, offset=offset_, gain=gain_)
    print("--- %s seconds for simple denoise CPU ---" % (time.time() - start_time))  # --salma

    ## smooth dead pixels
    win_ = 3
    #start_time = time.time()  # --salma
    imgDFiltered = daf.median_filter(imgD,  size=(1, win_, win_))
    imgDFiltered = imgDFiltered.compute() # - compute here before saving
    #imgDFiltered = median_filter(imgD,  size=(1, win_, win_))
    #print("--- %s seconds for median filter dask ---" % (time.time() - start_time))  # --salma

    # np.save(fishName+'/imgDNoMotion', imgD_)
    io.imsave(save_folder+'imgDNoMotion.tif', imgDFiltered, compress=1)

    t_ = len(imgDFiltered) // 2
    win_ = 150

    fix_ = imgDFiltered[t_ - win_:t_ + win_].mean(axis=0)
    #fix_ = da.mean(imgDFiltered[t_ - win_:t_ + win_], axis=0)
    #fix_ = dask.compute(fix_)
    np.save(save_folder + '/motion_fix_', fix_)


def pixel_denoise_img_seq_Tensor():
    from fish_proc.utils import getCameraInfo
    #from fish_proc.pixelwiseDenoising.simpleDenioseTool import simpleDN
    from scipy.ndimage.filters import median_filter
    from glob import glob
    from skimage import io
    import time  # --salma

    cameraInfo = getCameraInfo.getCameraInfo(root_folder)
    pixel_x0, pixel_x1, pixel_y0, pixel_y1 = [int(_) for _ in cameraInfo['camera_roi'].split('_')]
    pixel_x = (pixel_x0, pixel_x1)
    pixel_y = (pixel_y0, pixel_y1)

    imgStack = load_img_seq_dask().compute()  # Compute here
    #imgStack = load_img_seq()

    offset = np.load(cameraNoiseMat +'/offset_mat.npy').astype('float32')
    gain = np.load(cameraNoiseMat +'/gain_mat.npy').astype('float32')

    offset_ = offset[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    gain_ = gain[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]

    ## use tensor
    start_time = time.time()
    imgD = simpleDNTensor(imgStack, offset=offset_, gain=gain_)
    print("--- %s seconds for simple denoise GPU ---" % (time.time() - start_time))  # --salma
    print(imgD.shape, type(imgD))

    ## smooth dead pixels
    win_ = 3
    #start_time = time.time()  # --salma
    # imgDFiltered = daf.median_filter(imgD,  size=(1, win_, win_))
    # imgDFiltered = imgDFiltered.compute() # - compute here before saving
    imgDFiltered = median_filter(imgD,  size=(1, win_, win_))
    #print("--- %s seconds for median filter dask ---" % (time.time() - start_time))  # --salma

    # np.save(fishName+'/imgDNoMotion', imgD_)
    io.imsave(save_folder+'imgDNoMotion.tif', imgDFiltered, compress=1)

    t_ = len(imgDFiltered) // 2
    win_ = 150

    fix_ = imgDFiltered[t_ - win_:t_ + win_].mean(axis=0)
    #fix_ = da.mean(imgDFiltered[t_ - win_:t_ + win_], axis=0)

    np.save(save_folder + '/motion_fix_', fix_)

## python Dask_load_image_test
if __name__ == '__main__':
    if len(sys.argv)>1:
        ext = ''
        if len(sys.argv)>2:
            ext = sys.argv[2]
        eval(sys.argv[1]+f"({ext})")
    else:
        start_time = time.time()
        #pixel_denoise_img_seq()
        pixel_denoise_img_seq_Tensor()
        print("--- %s seconds for pixel_denoise: tensor ---" % (time.time() - start_time))