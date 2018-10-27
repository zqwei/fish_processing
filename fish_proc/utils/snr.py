import numpy as np
import cv2
import scipy.ndimage.filters as filters
from scipy.ndimage.filters import convolve

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

#import caiman as cm
import trefide.extras.tools as tools_
import trefide.extras.tool_grid as tgrid

import trefide.extras.denoise as denoise
# update noise estimators
# clean up comments after revision


def correlation_pnr(Y,
                    gSig=None, #deprecated
                    center_psf=True,
                    remove_small_val =False,
                    remove_small_val_th =3
                   ):
                    #swap_dim=True):
    """
    compute the correlation image and the peak-to-noise ratio (PNR) image.
    If gSig is provided, then spatially filtered the video.

    Args:
        Y:  np.ndarray (3D or 4D).
            Input movie data in 3D or 4D format
        gSig:  scalar or vector.
            gaussian width. If gSig == None, no spatial filtering
        center_psf: Boolearn
            True indicates subtracting the mean of the filtering kernel
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

    Returns:
        cn: np.ndarray (2D or 3D).
            local correlation image of the spatially filtered (or not)
            data
        pnr: np.ndarray (2D or 3D).
            peak-to-noise ratios of all pixels/voxels

    """
    #if swap_dim:
    #    Y = np.transpose(
    #        Y, tuple(np.hstack((Y.ndim - 1,
    #            list(range(Y.ndim))[:-1]))))

    # parameters
    #_, d1, d2 = Y.shape


    #data_raw = Y.reshape(-1, d1, d2).astype('float32')

    # filter data
    #data_filtered = data_raw.copy()
    #if gSig:
    #    if not isinstance(gSig, list):
    #        gSig = [gSig, gSig]
    #    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig])
        # create a spatial filter for removing background
        # psf = gen_filter_kernel(width=ksize, sigma=gSig, center=center_psf)

    #    if center_psf:
    #        for idx, img in enumerate(data_filtered):
    #            data_filtered[idx, ] = cv2.GaussianBlur(img,
    #                                                    ksize=ksize,
    #                                                    sigmaX=gSig[0],
    #                                                    sigmaY=gSig[1],
    #                                                    borderType=1) \
    #                - cv2.boxFilter(img, ddepth=-1, ksize=ksize, borderType=1)
    #        # data_filtered[idx, ] = cv2.filter2D(img, -1, psf, borderType=1)
    #    else:
    #        for idx, img in enumerate(data_filtered):
    #            data_filtered[idx, ] = cv2.GaussianBlur(
    #                img, ksize=ksize, sigmaX=gSig[0], sigmaY=gSig[1], borderType=1)

    # compute peak-to-noise ratio
    #data_filtered -= np.mean(data_filtered, axis=0)
    Y = Y - Y.mean(2,keepdims=True)
    #data_max = np.max(data_filtered, axis=0)
    data_max = Y.max(2)#,keepdims=True)
    #data_std = noise_estimator.get_noise_fft(data_filtered.transpose())[0].transpose()
    #data_std = noise_estimator.get_noise_fft(Y)[0]
    data_std = denoise.noise_level(Y)#[0]
    # Update to match noise from denoise.py here

    ## data_std = get_noise(data_filtered, method='diff2_med')
    pnr = np.divide(data_max, data_std)
    if remove_small_val:
        pnr[pnr < 0] = 0

    tmp_data = Y / data_std[:,:,np.newaxis]
    # remove small values
    #tmp_data = data_filtered.copy() / data_std
    if remove_small_val:
        tmp_data[tmp_data < remove_small_val_th] = 0

    # compute correlation image
    # cn = local_correlation(tmp_data, d1=d1, d2=d2)
    #cn = local_correlations_fft(tmp_data, swap_dim=False)
    cn = local_correlations_fft(tmp_data, swap_dim=True)

    return cn, pnr


def local_correlations_fft(Y,
                            eight_neighbours=True,
                            swap_dim=True,
                            opencv=True):
    """Computes the correlation image for the input dataset Y using a faster FFT based method

    Parameters:
    -----------

    Y:  np.ndarray (3D or 4D)
        Input movie data in 3D or 4D format

    eight_neighbours: Boolean
        Use 8 neighbors if true, and 4 if false for 3D data (default = True)
        Use 6 neighbors for 4D data, irrespectively

    swap_dim: Boolean
        True indicates that time is listed in the last axis of Y (matlab format)
        and moves it in the front

    opencv: Boolean
        If True process using open cv method

    Returns:
    --------

    Cn: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels

    """

    if swap_dim:
        Y = np.transpose(
            Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    Y = Y.astype('float32')
    Y -= np.mean(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    Ystd[Ystd == 0] = np.inf
    Y /= Ystd

    if Y.ndim == 4:
        if eight_neighbours:
            sz = np.ones((3, 3, 3), dtype='float32')
            sz[1, 1, 1] = 0
        else:
            sz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype='float32')
    else:
        if eight_neighbours:
            sz = np.ones((3, 3), dtype='float32')
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='float32')

    if opencv and Y.ndim == 3:
        Yconv = Y.copy()
        for idx, img in enumerate(Yconv):
            Yconv[idx] = cv2.filter2D(img, -1, sz, borderType=0)
        MASK = cv2.filter2D(
            np.ones(Y.shape[1:], dtype='float32'), -1, sz, borderType=0)
    else:
        Yconv = convolve(Y, sz[np.newaxis, :], mode='constant')
        MASK = convolve(
            np.ones(Y.shape[1:], dtype='float32'), sz, mode='constant')
    Cn = np.mean(Yconv * Y, axis=0) / MASK
    return Cn
