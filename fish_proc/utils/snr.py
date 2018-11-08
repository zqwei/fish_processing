import numpy as np

def correlation_pnr(Y, remove_small_val =False, remove_small_val_th =3, skip_pnr=False, skip_cn=False):
    from .np_mp import parallel_to_chunks
    Y = Y - Y.mean(axis=-1,keepdims=True)
    data_max = Y.max(axis=-1)
    cn, pnr = (None, None)
    if not skip_pnr:
        data_std, = parallel_to_chunks(noise_level, Y)
        pnr = np.divide(data_max, data_std)
        pnr[data_std==0]=0
        if remove_small_val:
            pnr[pnr < 0] = 0
    if not skip_cn:
        if remove_small_val:
            Y = Y / data_std[:,:,np.newaxis]
            Y[Y < remove_small_val_th] = 0
        cn = local_correlations_fft(Y, swap_dim=True)
    return cn, pnr

def noise_level(mov_wf, range_ff =[0.25,0.5]):
    # this is used for multiprocessing
    from . import noise_estimator
    ndim_ = np.ndim(mov_wf)
    if ndim_==3:
        dims_ = mov_wf.shape
        mov_wf = mov_wf.reshape((np.prod(dims_[:2]), dims_[2]),order='F')
    noise_level = noise_estimator.noise_estimator(mov_wf, range_ff=range_ff, method='logmexp')
    if ndim_ ==3:
        noise_level = noise_level.reshape(dims_[:2], order='F')
    return noise_level,


def local_correlations_fft(Y, eight_neighbours=True, swap_dim=True, opencv=True):
    from .np_mp import parallel_to_chunks
    from scipy.ndimage.filters import convolve
    if swap_dim:
        Y = np.transpose(
            Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    Y = Y.astype('float32')
    Y -= np.mean(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    Ystd[Ystd == 0] = np.inf
    Y /= Ystd
    len_=Y.shape[0]

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

    sum_, = parallel_to_chunks(local_correlations_fft_slice_sum, Y, sz=sz, opencv=opencv, ndim=Y.ndim)

    MASK = convolve(np.ones(Y.shape[1:], dtype='float32'), sz, mode='constant')
    # print(sum_.shape)
    Cn = sum_.sum(axis=0)/MASK/len_
    return Cn


def local_correlations_fft_slice_sum(imgs, sz=np.ones((3,3)), opencv=True, ndim=3):
    import cv2
    from scipy.ndimage.filters import convolve
    sum_ = np.zeros(imgs.shape[1:])
    for img in imgs:
        if opencv and ndim==3:
            sum_ += cv2.filter2D(img, -1, sz, borderType=0)*img
        else:
            sum_ += convolve(Y, sz[np.newaxis, :], mode='constant')*img
    return sum_[np.newaxis, :, :],
