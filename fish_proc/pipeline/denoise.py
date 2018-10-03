import numpy as np

def detrend(Y_, fishName, n_split = 1):
    from ..denoiseLocalPCA.detrend import trend
    from ..utils.np_mp import parallel_to_chunks
    from ..utils.memory import get_process_memory, clear_variables
    
    Y_trend = []
    for Y_split in np.array_split(Y_, n_split, axis=0):
        Y_trend.append(parallel_to_chunks(trend, Y_split.astype('float32'))[0].astype('float32'))
    
    # Y_trend = parallel_to_chunks(trend, Y_split)
    # Y_trend = Y_trend[0]
    # Y_trend_ = tuple([_ for _ in Y_trend])
    # Y_trend_ = np.concatenate(Y_trend_, axis=0)
    Y_trend = np.concatenate(Y_trend, axis=0).astype('float32')
    # Y_d = Y_ - Y_trend
    np.save(f'{fishName}/Y_d', Y_ - Y_trend)
    np.save(f'{fishName}/Y_trend', Y_trend)
    # Y_d = None
    Y_split = None
    Y_trend = None
    clear_variables((Y_split, Y_, Y_trend))
    get_process_memory();
    return None

def denose_2dsvd(Y_d, fishName, nblocks=[10, 10], stim_knots=None, stim_delta=0):
    from ..denoiseLocalPCA.denoise import temporal as svd_patch
    from ..utils.memory import get_process_memory, clear_variables
    
    Y_d_ave = Y_d.mean(axis=-1, keepdims=True) # remove mean
    Y_d_std = Y_d.std(axis=-1, keepdims=True) # normalization
    Y_d = (Y_d - Y_d_ave)/Y_d_std
    Y_d = Y_d.astype('float32')
    np.savez(f'{fishName}/Y_2dnorm', Y_d_ave=Y_d_ave, Y_d_std=Y_d_std)
    Y_d_ave = None
    Y_d_std = None
    clear_variables((Y_d_ave, Y_d_std))
    get_process_memory();
    
    
    dx=4
    maxlag=5
    confidence=0.99
    greedy=False,
    fudge_factor=1
    mean_th_factor=1.15
    U_update=False
    min_rank=1
    
    Y_svd, _ = svd_patch(Y_d, nblocks=nblocks, dx=dx, stim_knots=stim_knots, stim_delta=stim_delta)
    np.save(f'{fishName}/Y_2dsvd', Y_svd)
    return None
