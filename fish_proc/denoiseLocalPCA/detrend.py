from .preprocess import _get_spline_trend
import numpy as np
from scipy.signal import butter, lfilter

def detrend(mov,
            stim=None,
            disc_idx=None,
            order=3,
            followup=100,
            spacing=2000,
            q=.05,
            axis=-1,
            robust=True):
    # Adaptive spline fit

    if stim is None:
        stim = np.zeros(mov.shape[axis])

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


def trend(mov, order=3, followup=100, spacing=2000, q=.05, axis=-1, robust=True):
    # Adaptive spline fit
    stim = np.zeros(mov.shape[axis])
    return _get_spline_trend(data=mov, stim=stim, disc_idx=None, order=order,
                             followup=followup, spacing=spacing, q=q, axis=axis, robust=robust),


def trend_sg(mov, win_=201, order=7):
    from scipy.signal import savgol_filter
    trend = savgol_filter(Y, win_, order, axis=-1)
    return trend

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
