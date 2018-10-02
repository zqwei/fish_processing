"""
A simplified version of denoising tool

@author: modified by Ziqiang Wei, Jan 2018

"""

import numpy as np
from glob import glob
from os.path import exists
# from ..utils import np_mp # this should be fixed later

def compute_gain_matrix(folder_name):
    offsetMat = [];
    varmat = [];
    for nfile in sorted(glob(folder_name + '/*.npz')):
        print(nfile)
        data = np.load(nfile)
        tOff = data['arr_0']
        tvar = data['arr_1']
        offsetMat.append(tOff)
        varmat.append(tvar)
    offsetMat = np.array(offsetMat)
    varmat = np.array(varmat)
    gainMatShape = (varmat.shape[1], varmat.shape[2])
    # image to vector
    offsetVec = offsetMat.reshape(offsetMat.shape[0], -1)
    varVec = varmat.reshape(varmat.shape[0], -1)
    # remove 0mW (background)
    offsetVec = offsetVec - offsetVec[0]
    varVec = varVec - varVec[0]
    # remove 0mW
    offsetVec = offsetVec[1:]
    varVec = varVec[1:]
    gainVec = np.zeros(offsetVec.shape[1])
    for n in range(offsetVec.shape[1]):
        noffset = offsetVec[:, n]
        nvar = varVec[:, n]
        gainVec[n] = np.inner(noffset, nvar)/ np.inner(noffset, noffset)
        if np.sum(nvar<0)>0 or gainVec[0]<0:
            gainVec[n] = 0

    offset = offsetMat[0]
    var = varmat[0]
    gain = np.array(gainVec).reshape(gainMatShape)

    np.save(folder_name +'/offset_mat.npy', offset)
    np.save(folder_name +'/var_mat.npy', var)
    np.save(folder_name +'/gain_mat.npy', gain)

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

    # compute processesed image
    imgD = (img - offset_) / (gain_ + 1e-12)
    imgD[np.broadcast_to(gain_ < 0.5, imgD.shape)] = 1e-6
    imgD[np.broadcast_to(gain_ > 5, imgD.shape)] = 1e-6
    imgD[imgD <= 0] = 1e-6
    return imgD

def smoothDeadPixelBoxCar(img):
    from scipy.signal import convolve2d
    win_size = 1
    box_blur = np.ones((win_size*2+1, win_size*2+1))
    box_blur[win_size, win_size] = 0
    box_blur = box_blur/box_blur.sum()   
    imgD = np.empty(img.shape)
    for n_ in range(img.shape[0]):
        img_ = img[n_].copy()
        box_img = convolve2d(img_, box_blur, boundary='symm', mode='same')
        min_ = box_img.min()*100
        img_[img_<min_] = box_img[img_<min_]
        imgD[n_] = img_
    return imgD

# def simpleDNStack(img, folder_name='../pixelwiseDenoising/gainMat20180208', pixel_x=None, pixel_y=None, offset=None, gain=None):
#     imgD = img.copy()
#     for nPlane in range(len(imgD)):
#         img_ = img[nPlane]
#         imgD[nPlane] = simpleDN(img_, folder_name=folder_name, pixel_x=pixel_x, pixel_y=pixel_y, offset=offset, gain=gain)
#     return imgD

# def simpleDNStackMP(img, folder_name='../pixelwiseDenoising/gainMat20180208', pixel_x=None, pixel_y=None, offset=None, gain=None):
#     imgD = img.copy()
#     return np_mp.parallel_apply_along_axis(simpleDN, 0, imgD, folder_name=folder_name, pixel_x=pixel_x, pixel_y=pixel_y, offset=offset, gain=gain)

if __name__ == '__main__':
    compute_gain_matrix(folder_name='../pixelwiseDenoising/gainMat20180208')
