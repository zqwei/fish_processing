from . import spatial_filtering
from . import tool_grid
import time
# ____________________________
# Wrapper to call denoisers
# see individual functions for more information
# ____________________________

def spatial(Y_new,
            gHalf=[2,2],
            sn=None):
    """
    Calls spatial wiener filter in pixel neighborhood
    """
    mov_wf,_ = spatial_filtering.spatial_filter_image(Y_new,
                                                gHalf=gHalf,
                                                sn=sn)
    return mov_wf


def temporal(W,
             nblocks=[10,10],
             dx=1,
             maxlag=5,
             confidence=0.99,
             greedy=False,
             fudge_factor=1,
             mean_th_factor=1.15,
             U_update=False,
             min_rank=1,
             stim_knots=None,
             stim_delta=200, is_single_core=False, mask=None):
    """
    Calls greedy temporal denoiser in pixel neighborhood
    """
    start = time.time()
    if mask is not None:
        W[mask.squeeze()] = 0
    print(W.dtype)
    mov_d, ranks = tool_grid.denoise_dx_tiles(W,
                                              nblocks=nblocks,
                                              dx=dx,
                                              maxlag=maxlag,
                                              confidence=confidence,
                                              greedy=greedy,
                                              fudge_factor=fudge_factor,
                                              mean_th_factor=mean_th_factor,
                                              U_update=U_update,
                                              min_rank=min_rank,
                                              stim_knots=stim_knots,
                                              stim_delta=stim_delta, is_single_core=is_single_core)
    print('Run_time: %f'%(time.time()-start), flush=True)
    return mov_d, ranks