import numpy as np
import pandas as pd
import os, sys
from glob import glob
from h5py import File
from fish_proc.utils.getCameraInfo import getCameraInfo

def pixelDenoiseImag(img, cameraNoiseMat='', cameraInfo=None):
    from fish_proc.pixelwiseDenoising.simpleDenioseTool import simpleDN
    from scipy.ndimage.filters import median_filter
    win_ = 3
    pixel_x0, pixel_x1, pixel_y0, pixel_y1 = [int(_) for _ in cameraInfo['camera_roi'].split('_')]
    pixel_x = (pixel_x0, pixel_x1)
    pixel_y = (pixel_y0, pixel_y1)
    offset = np.load(cameraNoiseMat +'/offset_mat.npy').astype('float32')
    gain = np.load(cameraNoiseMat +'/gain_mat.npy').astype('float32')
    offset_ = offset[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    gain_ = gain[pixel_x[0]:pixel_x[1], pixel_y[0]:pixel_y[1]]
    if img.ndim == 3:
        img = np.expand_dims(img, axis=1) # insert z dim here
    filter_win = (1, 1, win_, win_)
    return median_filter(simpleDN(img, offset=offset_, gain=gain_), size=filter_win)


def load_bz2file(file, dims):
    import bz2
    import numpy as np
    data = bz2.BZ2File(file,'rb').read()
    im = np.frombuffer(data,dtype='int16')
    return im.reshape(dims[-1::-1])


def estimate_rigid2d(moving, fixed=None, affs=None, to3=True):
    from fish_proc.imageRegistration.imTrans import ImAffine
    from numpy import expand_dims
    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.factors = [8, 4, 2]
    trans.sigmas = [3.0, 2.0, 1.0]
    trans.ss_sigma_factor = 1.0
    affs = trans.estimate_rigid2d(fixed.max(0), moving.squeeze(axis=0).max(0), tx_tr=affs).affine
    if to3:
        _ = np.eye(4)
        _[1:, 1:] = affs
        affs = _
    return expand_dims(affs, 0)


def rigid_interp(trans_affine, down_sample_registration, len_dat):
    trans_affine_ = np.repeat(trans_affine, down_sample_registration, axis=0)
    return trans_affine_[:len_dat]


def estimate_translation2d(moving, fixed=None, to3=True):
    from fish_proc.imageRegistration.imTrans import ImAffine
    from numpy import expand_dims
    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.ss_sigma_factor = 1.0
    trans.factors = [8, 4, 2]
    trans.sigmas = [3.0, 1.0, 1.0]
    affs = trans.estimate_translation2d(fixed.max(0), moving.squeeze().max(0)).affine
    if to3:
        _ = np.eye(4)
        _[1:, 1:] = affs
        affs = _
    return expand_dims(affs, 0)


def apply_transform3d(mov, affs):
    from scipy.ndimage.interpolation import affine_transform
    return np.expand_dims(affine_transform(mov.squeeze(axis=0), affs.squeeze(axis=0)), 0)


def save_h5(filename, data, dtype='float32'):
    with File(filename, 'w') as f:
        f.create_dataset('default', data=data.astype(dtype), compression='gzip', chunks=True, shuffle=True)
        f.close()


def save_h5_rescale(filename, data, reset_max_int=65535):
    ## np.iinfo(np.uint16).max = 65535
    with File(filename, 'w') as f:
        data_max = data.max()
        data_min = data.min()
        data = (data - data_min)/(data_max - data_min)*reset_max_int
        f.create_dataset('default', data=data.astype(np.uint16), compression='gzip', chunks=True, shuffle=True)
        f.create_dataset('scale', data=np.array([data_min, data_max]), chunks=True, shuffle=True)
        f.close()


def baseline(data, window=100, percentile=15, downsample=10, axis=-1):
    from scipy.ndimage.filters import percentile_filter
    from scipy.interpolate import interp1d
    from numpy import ones

    size = ones(data.ndim, dtype='int')
    size[axis] *= window//downsample

    if downsample == 1:
        bl = percentile_filter(data, percentile=percentile, size=size)
    else:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, None, downsample)
        data_ds = data[slices]
        baseline_ds = percentile_filter(data_ds, percentile=percentile, size=size)
        interper = interp1d(range(0, data.shape[axis], downsample), baseline_ds, axis=axis, fill_value='extrapolate')
        bl = interper(range(data.shape[axis]))
    return bl


def baseline_from_Yd(block_t, block_d):
    min_t = np.percentile(block_t, 0.3, axis=-1, keepdims=True)
    min_t[min_t>0] = 0
    return block_t - block_d - min_t


def baseline_correct(block_b, block_t):
    min_t = np.percentile(block_t, 0.3, axis=-1, keepdims=True)
    min_t[min_t>0] = 0
    min_b = np.min(block_b-min_t, axis=-1, keepdims=True)
    min_b[min_b<=0] = min_b[min_b<=0] - 0.01
    min_b[min_b>0] = 0
    return block_b - min_t - min_b


def robust_sp_trend(mov):
    from fish_proc.denoiseLocalPCA.detrend import trend
    return trend(mov)


def fb_pca_block(block, mask_block, block_id=None):
    # using fb pca instead of local pca from fish
    # from fbpca import pca
    from sklearn.utils.extmath import randomized_svd
    from numpy import expand_dims
    if mask_block.sum()==0:
        return np.zeros(block.shape)
    M = (block-block.mean(axis=-1, keepdims=True)).squeeze()
    M[~mask_block.squeeze()] = 0
    dimsM = M.shape
    M = M.reshape((np.prod(dimsM[:-1]),dimsM[-1]),order='F')
    k = min(min(M.shape)//4, 300)
    # [U, S, Va] = pca(M.T, k=k, n_iter=20, raw=True)
    [U, S, Va] = randomized_svd(M.T, k, n_iter=10, power_iteration_normalizer='QR')
    M_pca = U.dot(np.diag(S).dot(Va))
    M_pca = M_pca.T.reshape(dimsM, order='F')
    return expand_dims(M_pca, 0)


# mask functions
def intesity_mask(blocks, percentile=40):
    return blocks>np.percentile(blocks, percentile)


def intesity_mask_block(blocks, percentile):
    return blocks>np.percentile(blocks, percentile.squeeze())


def snr_mask(Y_svd, std_per=20, snr_per=10):
    Y_svd = Y_svd.squeeze()
    d1, d2, _ = Y_svd.shape
    mean_ = Y_svd.mean(axis=-1,keepdims=True)
    sn, _ = get_noise_fft(Y_svd - mean_,noise_method='logmexp')
    SNR_ = Y_svd.var(axis=-1)/sn**2
    Y_d_std = Y_svd.std(axis=-1)
    std_thres = np.percentile(Y_d_std.ravel(), std_per)
    mask = Y_d_std<=std_thres
    snr_thres = np.percentile(np.log(SNR_).ravel(), snr_per)
    mask = np.logical_or(mask, np.log(SNR_)<snr_thres)
    return mask.squeeze()


def mask_blocks(block, mask=None):
    if block.ndim != 4 or mask.ndim !=4:
        print('error in block shape or mask shape')
        return None
    _ = block.copy()
    _[~mask.squeeze(axis=-1)] = 0
    return _


def demix_blocks_square(block, mask_block, save_folder='.', is_skip=True, block_id=None):
    from skimage.exposure import equalize_adapthist as clahe
    from skimage.morphology import square, dilation
    from skimage.segmentation import watershed, felzenszwalb
    from sklearn.decomposition import NMF
    from skimage.filters import gaussian as f_gaussian
    from scipy.sparse import csr_matrix
    from skimage.measure import label as m_label

    # set fname for blocks
    fname = 'period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
    # set no processing conditions
    sup_fname = f'{save_folder}/sup_demix_rlt/'+fname

    block_img = mask_block.squeeze()
    if block_img.max()==0:
        np.savez(sup_fname+'_rlt.npz', A=np.zeros([np.prod(dims[:-1]),1]))
        return np.zeros([1]*4)
    mx, my = block_img.shape
    n_split = 4
    x_vec = np.array_split(np.arange(mx), n_split)
    y_vec = np.array_split(np.arange(my), n_split)
    block_list = []
    s = np.zeros((mx, my))
    for ny in y_vec:
        for nx in x_vec:
            s_ = s.copy()
            s_[nx.min():nx.max()+3,ny.min():ny.max()+3]=1
            block_list.append(s_.reshape(-1, order='F'))
    a_ini = np.array(block_list).astype('bool').T

    Yt = block.squeeze()
    dims = Yt.shape;
    T = dims[-1];
    Yt_r = Yt.reshape(np.prod(dims[:-1]),T,order = "F");
    Yt_r[Yt_r<0]=0
    Yt_r = csr_matrix(Yt_r);
    model = NMF(n_components=1, init='custom')
    U_mat = []
    V_mat = []
    if Yt_r.sum()==0:
        np.savez(sup_fname+'_rlt.npz', A=np.zeros([np.prod(dims[:-1]),1]))
        return np.zeros([1]*4)
    for ii, comp in enumerate(a_ini.T):
        y_temp = Yt_r[comp,:].astype('float')
        if y_temp.sum()==0:
            continue
        u_ = np.zeros((np.prod(dims[:-1]),1)).astype('float32')
        u_[list(comp)] = model.fit_transform(y_temp, W=np.array(y_temp.mean(axis=1)),H = np.array(y_temp.mean(axis=0)))
        U_mat.append(u_)
        V_mat.append(model.components_.T)
    if len(U_mat)>1:
        U_mat = np.concatenate(U_mat, axis=1)
        V_mat = np.concatenate(V_mat, axis=1)
    else:
        U_mat = np.zeros([np.prod(dims[:-1]),1])
        V_mat = np.zeros([T,1])

    if U_mat.sum()>0:
        model_ = NMF(n_components=U_mat.shape[-1], init='custom', solver='cd', max_iter=20)
        U = model_.fit_transform(Yt_r.astype('float'), W=U_mat.astype('float64'), H=V_mat.T.astype('float64'))
        V = model_.components_
    else:
        U = np.zeros([np.prod(dims[:-1]),1])
    # clean up the NMF results
    U[U<U.max(axis=-1, keepdims=True)*.3]=0
    # split components
    U_ext = []
    V_ext = []
    for n_u in range(U.shape[1]):
        u_ = U[:, n_u].reshape(mx, my, order='F')
        u_blur = f_gaussian(u_, sigma=1/12)
        blobs_labels = m_label(u_blur>u_blur.max()*0.3, background=0)
        n_max = blobs_labels.max()
        for n_b in range(1, n_max+1):
            if(blobs_labels==n_b).sum()>10:
                u_b = u_.copy()
                u_b[blobs_labels!=n_b] = 0
                U_ext.append(u_b.reshape(-1, order='F'))
                V_ext.append(V[n_u])
    if len(U_ext)==0:
        np.savez(sup_fname+'_rlt.npz', A=np.zeros([np.prod(dims[:-1]),1]))
        return np.zeros([1]*4)

    U_ext = np.array(U_ext, order='F').T
    V_ext = np.array(V_ext, order='F').T
    model_ = NMF(n_components=U_ext.shape[-1], init='custom', solver='mu', max_iter=10)
    U = model_.fit_transform(Yt_r.astype('float'), W=U_ext.astype('float64'), H=V_ext.T.astype('float64'))
    U[U<U.max(axis=-1, keepdims=True)*.3]=0
    temp = np.sqrt((U**2).sum(axis=0,keepdims=True))
    U = U/temp
    np.savez(sup_fname+'_rlt.npz', A=U)
    return np.zeros([1]*4)


def demix_blocks(block, mask_block, save_folder='.', is_skip=True, block_id=None):
    from skimage.exposure import equalize_adapthist as clahe
    from skimage.morphology import square, dilation
    from skimage.segmentation import watershed, felzenszwalb
    from sklearn.decomposition import NMF
    from skimage.filters import gaussian as f_gaussian
    from scipy.sparse import csr_matrix
    from skimage.measure import label as m_label

    # set fname for blocks
    fname = 'period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
    # set no processing conditions
    sup_fname = f'{save_folder}/sup_demix_rlt/'+fname

    block_img = mask_block.squeeze()
    if block_img.max()==0:
        np.savez(sup_fname+'_rlt.npz', A=np.zeros([np.prod(dims[:-1]),1]))
        return np.zeros([1]*4)
    mx, my = block_img.shape
#     try:
#         img_adapteq = clahe(block_img/block_img.max(), clip_limit=0.03)
#     except:
#         img_adapteq = block_img
    img_adapteq = block_img/block_img.max()
    # initial segments
    # segments_watershed = watershed(img_adapteq, markers=20, compactness=0.01)
    segments_watershed = felzenszwalb(img_adapteq, scale=10, sigma=0.01, min_size=50)
    min_ = segments_watershed.min()
    max_ = segments_watershed.max()
    vec_segments_watershed = segments_watershed.reshape(-1, 1, order='F')
    a_ini = np.zeros((vec_segments_watershed.shape[0], max_+1))
    for n in range(min_, max_+1):
        if (vec_segments_watershed[:,0]==n).sum()>10:
            _ = (segments_watershed==n)
            _ = dilation(_, square(5)).astype('float')
            _ = _.reshape(-1, order='F')
            a_ini[_>0, n]=1
    a_ini = a_ini[:, a_ini.sum(0)>0]>0

    Yt = block.squeeze()
    dims = Yt.shape;
    T = dims[-1];
    Yt_r = Yt.reshape(np.prod(dims[:-1]),T,order = "F");
    Yt_r[Yt_r<0]=0
    Yt_r = csr_matrix(Yt_r);
    model = NMF(n_components=1, init='custom')
    U_mat = []
    V_mat = []
    if Yt_r.sum()==0:
        np.savez(sup_fname+'_rlt.npz', A=np.zeros([np.prod(dims[:-1]),1]))
        return np.zeros([1]*4)
    for ii, comp in enumerate(a_ini.T):
        y_temp = Yt_r[comp,:].astype('float')
        if y_temp.sum()==0:
            continue
        u_ = np.zeros((np.prod(dims[:-1]),1)).astype('float32')
        u_[list(comp)] = model.fit_transform(y_temp, W=np.array(y_temp.mean(axis=1)),H = np.array(y_temp.mean(axis=0)))
        U_mat.append(u_)
        V_mat.append(model.components_.T)
    if len(U_mat)>1:
        U_mat = np.concatenate(U_mat, axis=1)
        V_mat = np.concatenate(V_mat, axis=1)
    else:
        U_mat = np.zeros([np.prod(dims[:-1]),1])
        V_mat = np.zeros([T,1])

    if U_mat.sum()>0:
        model_ = NMF(n_components=U_mat.shape[-1], init='custom', solver='cd', max_iter=20)
        U = model_.fit_transform(Yt_r.astype('float'), W=U_mat.astype('float64'), H=V_mat.T.astype('float64'))
        V = model_.components_
    else:
        U = np.zeros([np.prod(dims[:-1]),1])
    # clean up the NMF results
    U[U<U.max(axis=-1, keepdims=True)*.3]=0
    # split components
    U_ext = []
    V_ext = []
    for n_u in range(U.shape[1]):
        u_ = U[:, n_u].reshape(mx, my, order='F')
        u_blur = f_gaussian(u_, sigma=1/12)
        blobs_labels = m_label(u_blur>u_blur.max()*0.3, background=0)
        n_max = blobs_labels.max()
        for n_b in range(1, n_max+1):
            if(blobs_labels==n_b).sum()>10:
                u_b = u_.copy()
                u_b[blobs_labels!=n_b] = 0
                U_ext.append(u_b.reshape(-1, order='F'))
                V_ext.append(V[n_u])
    if len(U_ext)==0:
        np.savez(sup_fname+'_rlt.npz', A=np.zeros([np.prod(dims[:-1]),1]))
        return np.zeros([1]*4)

    U_ext = np.array(U_ext, order='F').T
    V_ext = np.array(V_ext, order='F').T
    model_ = NMF(n_components=U_ext.shape[-1], init='custom', solver='mu', max_iter=10)
    U = model_.fit_transform(Yt_r.astype('float'), W=U_ext.astype('float64'), H=V_ext.T.astype('float64'))
    U[U<U.max(axis=-1, keepdims=True)*.3]=0
    temp = np.sqrt((U**2).sum(axis=0,keepdims=True))
    U = U/temp
    np.savez(sup_fname+'_rlt.npz', A=U)
    return np.zeros([1]*4)


def demix_file_name_block(save_root='.', ext='', block_id=None):
    fname = f'{save_root}/demix_rlt{ext}/period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
    return fname+'_rlt.pkl'


def sup_file_name_block(save_root='.', ext='', block_id=None):
    fname = f'{save_root}/sup_demix_rlt{ext}/period_Y_demix_block_'
    for _ in block_id:
        fname += '_'+str(_)
    return fname+'_rlt.npz'


def load_A_matrix(save_root='.', ext='', block_id=None, min_size=40):
    fname = sup_file_name_block(save_root=save_root, ext=ext, block_id=block_id)
    _ = np.load(fname, allow_pickle=True)
    return _['A']


def pos_sig_correction(mov, dt, axis_=-1):
    return mov - (mov[:, :, dt]).min(axis=axis_, keepdims=True)


def compute_cell_raw_dff(block_F, mask, save_root='.', ext='', block_id=None):
    _, x_, y_, _ = block_F.shape
    A_= load_A_matrix(save_root=save_root, ext=ext, block_id=block_id, min_size=0)

    if A_.sum()==0:
        return np.zeros([1]*4) # return if no components

    if mask.sum()==0:
        return np.zeros([1]*4) # return if out of brain

    fsave = f'{save_root}/cell_raw_dff/period_Y_demix_block_'
    for _ in block_id:
        fsave += '_'+str(_)
    fsave += '_rlt.h5'

    A_ = A_[:, A_.sum(axis=0)>0] # remove zero-components
    F_ = block_F.squeeze(axis=0).reshape((x_*y_, -1), order='F')
    cell_F = np.linalg.inv(A_.T.dot(A_)).dot(np.matmul(A_.T, F_)) #demix from inversion

    A_[A_<A_.max(axis=0, keepdims=True)*0.3]=0
    A_ = A_.reshape((x_, y_, -1), order="F")

    mask_ = mask.squeeze()
    A_sparse = A_.copy()
    A_sparse[~mask_]=0
    valid = (A_sparse>0).sum(axis=(0,1))>10
    if valid.sum()>0:
        A_sparse = A_sparse[:, :, (A_sparse>0).sum(axis=(0,1))>10]
        F_sparse = block_F.squeeze(axis=0)
        F_sparse[~mask_]=0
        F_sparse = F_sparse.reshape((x_*y_, -1), order='F')
        A__ = A_sparse.copy()
        A__ = A__.reshape((x_*y_, -1), order='F')
        cell_F_sparse = np.linalg.inv(A__.T.dot(A__)).dot(np.matmul(A__.T, F_sparse))


    with File(fsave, 'w') as f:
        f.create_dataset('A_loc', data=np.array([block_id[0], x_*block_id[1], y_*block_id[2]]))
        f.create_dataset('A', data=A_)
        f.create_dataset('cell_F', data=cell_F)
        if valid.sum()>0:
            f.create_dataset('A_s', data=A_sparse)
            f.create_dataset('cell_F_s', data=cell_F_sparse)
        f.close()
    return np.zeros([1]*4)
