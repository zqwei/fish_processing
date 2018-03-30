"""
A simplified version of denoising tool

@author: modified by Ziqiang Wei, Jan 2018

"""

import numpy as np
from glob import glob
from os.path import exists

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
    if pixel_x is None:
        pixel_x = (0, img.shape[0])
    if pixel_y is None:
        pixel_y = (0, img.shape[1])

    # load gain and offset matrix
    if offset is None:
        assert exists(folder_name)
        offset = np.load(folder_name +'/offset_mat.npy')
    if gain is None:
        gain = np.load(folder_name +'/gain_mat.npy')
    offset_ = offset[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    gain_ = gain[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]

    # compute processesed image
    imgD = (img - offset_) / (gain_ + 1e-12)
    imgD[gain_ < 0.5] = 1e-6
    imgD[imgD <= 0] = 1e-6
    return imgD

def simpleDNStack(img, folder_name='../pixelwiseDenoising/gainMat20180208', pixel_x=None, pixel_y=None, offset=None, gain=None):
    imgD = img.copy()
    for nPlane in range(len(imgD)):
        img_ = img[nPlane]
        imgD[nPlane] = simpleDN(img_, folder_name=folder_name, pixel_x=pixel_x, pixel_y=pixel_y, offset=offset, gain=gain)
    return imgD

if __name__ == '__main__':
    compute_gain_matrix(folder_name='../pixelwiseDenoising/gainMat20180208')
