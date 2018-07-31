from trefide.preprocess import _get_spline_trend
import numpy as np

def detrend(mov,
            stim=None,
            disc_idx=None,
            order=3,
            followup=100,
            spacing=200,
            q=.05,
            axis=-1,
            robust=True):
    # Adaptive spline fit

    if stim is None:
        stim = np.zeros(mov.shape[-1])

    trend = _get_spline_trend(data=mov,
                              stim=stim,
                              disc_idx=disc_idx,
                              order=order,
                              followup=followup,
                              spacing=spacing,
                              q=q,
                              axis=axis,
                              robust=robust)

    mov_detr = np.subtract(mov, trend)
    # Remove samples from discontinuity locations
    if disc_idx is not None:
        del_idx = np.sort(np.append(np.append(disc_idx, disc_idx + 1),
                                    disc_idx - 1))
        stim = np.delete(stim, del_idx)
        mov_detr = np.delete(mov_detr, del_idx, axis=-1)
        trend = np.delete(trend, del_idx, axis=-1)
        # Recompute problem areas
        disc_idx[1:] = disc_idx[1:] - np.cumsum(np.ones(len(disc_idx) - 1) * 3)
        disc_idx = disc_idx - 1
        disc_idx = np.append(disc_idx,
                             np.argwhere(filt.convolve1d(stim > 0,
                                                         np.array([1, -1]))))
    return mov_detr, trend, stim, np.unique(disc_idx)


def detrend_sg(mov, win_=201, order=7):
    from scipy.signal import savgol_filter
    trend = savgol_filter(Y, win_, order, axis=-1)
    return Y - trend, trend
