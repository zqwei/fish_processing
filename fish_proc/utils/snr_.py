import numpy as np
import cv2
from scipy.ndimage.filters import convolve

def correlation_pnr(Y,
                    gSig=None, #deprecated
                    center_psf=True,
                    remove_small_val =False,
                    remove_small_val_th =3
                   ):
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
    from .np_mp import parallel_to_chunks
    Y = Y - Y.mean(2,keepdims=True)
    data_max = Y.max(2)#,keepdims=True)
    # data_std = noise_level(Y)
    data_std, = parallel_to_chunks(noise_level, Y)
    pnr = np.divide(data_max, data_std)
    pnr[data_std==0]=0
    if remove_small_val:
        pnr[pnr < 0] = 0
    if remove_small_val:
        Y = Y / data_std[:,:,np.newaxis]
        Y[Y < remove_small_val_th] = 0
    cn = local_correlations_fft(Y, swap_dim=True)
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

def noise_level(mov_wf,
                range_ff =[0.25,0.5]):
    """
    Calculate noise level in movie pixels
    """
    from . import noise_estimator

    ndim_ = np.ndim(mov_wf)
    if ndim_==3:
        dims_ = mov_wf.shape
        mov_wf = mov_wf.reshape((np.prod(dims_[:2]), dims_[2]),order='F')
    noise_level = noise_estimator.noise_estimator(mov_wf, range_ff=range_ff,
                                                  method='logmexp')
    if ndim_ ==3:
        noise_level = noise_level.reshape(dims_[:2], order='F')

    return noise_level,
