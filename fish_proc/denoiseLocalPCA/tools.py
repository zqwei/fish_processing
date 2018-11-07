import numpy as np
import scipy as sp
import scipy.signal as sps
import scipy.ndimage as ndi
import scipy.sparse as spr


def reshape_dims(M,dims=None):

    num_dim = np.ndim(M)
    if num_dim ==3:
        M1r= M.reshape((np.prod(dims[:2]),dims[2]),order='F')
    elif num_dim ==2:
        M1r = M.reshape(dims,order='F')
    return M1r

def remove_trend(Y_rm,detrend_option='linear'):
    mean_pixel = Y_rm.mean(axis=1, keepdims=True)
    Y_rm2 = Y_rm - mean_pixel
    # Detrend
    if detrend_option=='linear':
        detr_data = sp.signal.detrend(Y_rm2,axis=1,type='l')
    #elif detrend_option=='quad':
        #detr_data = detrend(Y_rm)
    else:
        print('Add option')
    Y_det = detr_data + mean_pixel
    offset = Y_rm - Y_det
    return Y_det, offset


def unpad(x):
    """
    Given padded matrix with nan
    Get rid of all nan in order (row, col)

    Parameters:
    ----------
    x:          np.array
                array to unpad (all nan values)

    Outputs:
    -------
    x:          np.array
                unpaded array (will not contain nan values)
                dimension might be different from input array
    """
    x = x[:, ~np.isnan(x).all(0)]
    x = x[~np.isnan(x).all(1)]
    return x


def pad(array, reference_shape, offsets, array_type=np.nan):
    """
    Pad array wrt reference_shape exlcluding offsets with dtype=array_type

    Parameters:
    ----------
    array:          np.array
                    array to be padded
    reference_shape:tuple
                    size of narray to create
    offsets:        tuple
                    list of offsets (number of elements must be equal
                    to the dimension of the array)
                    will throw a ValueError if offsets is too big and the
                    reference_shape cannot handle the offsets
    array_type:     dtype
                    data type to pad array with.

    Outputs:
    -------
    result:         np.array (reference_shape)
                    padded array given input
    """

    # Create an array of zeros with the reference shape
    result = np.ones(reference_shape) * array_type
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim])
                  for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result

def nextpow2(value):
    """
    Find exponent such that 2^exponent is >= abs(value).

    Parameters:
    ----------
    value : int

    Returns:
    -------
    exponent : int
    """

    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent


def axcov(data, maxlag=10):
    """
    Edited from cnmf.deconvolution
    Compute the autocovariance of data at lag = -maxlag:0:maxlag

    Parameters:
    ----------
    data : array
        Array containing fluorescence data

    maxlag : int
        Number of lags to use in autocovariance calculation

    Output:
    -------
    axcov : array
        Autocovariances computed from -maxlag:0:maxlag
    """

    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    return np.real(np.divide(xcov, T))


#### SOME FILTERS


def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.

    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def FIR_filter(x,
               sample_rate,
               transition_width=5.0,
               ripple_db=60,
               cutoff_hz=10.0
              ):
    from scipy.signal import kaiserord, lfilter, firwin, freqz

    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = transition_width/nyq_rate

    # The desired attenuation in the stop band, in dB.
    #ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = cutoff_hz

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

    # Use lfilter to filter x with the FIR filter.
    filtered_x = lfilter(taps, 1.0, x)

    return filtered_x


def background_noise():
    # we same background is shared across all components
    # estimate only-noise signal through tresholds
    # estimate threshold
    # threshold signals
    # from any pixel which doesn't have spikes
    # run rank 1-2 svd and estimate the background from more than 2 pixels

    return

def butter_highpass(signals, cutoff, fs, order=6, axis=-1):
    """ Forward-backward filter inpute signals with butterworth kernel"""
    return sps.filtfilt(*sps.butter(order,
                                    cutoff / (0.5 * fs),
                                    btype='high',
                                    analog=False),
                        signals,
                        axis=axis)


def gaussian_bandpass(signals, kernel_len, kernel_sigma, axis=-1):
    """ Convolve inputs signals with 0 mean gaussian kernel"""
    return ndi.convolve1d(signals,
                          gaussian_kernel(kernel_len,
                                          kernel_sigma),
                          axis=axis)


def gaussian_kernel(length, sigma):
    """
    Returns a 1D gaussian filter of specified length and stdv for convolution
    """
    n = (length - 1.) / 2.
    kernel = np.exp((-1 * np.power(np.arange(-n, n + 1), 2)) / (2. * sigma**2))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    sumh = kernel.sum()
    if sumh != 0:
        kernel /= sumh
    return kernel - kernel.mean()


def detrend(signals, degree=2):
    """ Substracts the order k trend from each input pixel """
    _, len_signal = signals.shape
    X = np.array([np.power(np.arange(len_signal), k)
                  for k in range(degree + 1)]).T
    Hat = X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
    return signals - signals.dot(Hat)


def HALS4activity(Yr, A, C, K, nb, iters=2):
    U = A.T.dot(Yr)
    V = A.T.dot(A)
    for _ in range(iters):
        for m in range(K):  # neurons
            C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) /
                           V[m, m], -np.inf, np.inf)
        for m in range(nb):  # background
            C[K + m] = np.clip(C[K + m] + (U[K + m] - V[K + m].dot(C)) /
                               V[K + m, K + m], -np.inf, np.inf)
    return C


def HALS4shape(Yr, A, C, K, nb, ind_A, iters=2):
    U = C.dot(Yr.T)
    V = C.dot(C.T)
    for _ in range(iters):
        for m in range(K):  # neurons
            ind_pixels = np.squeeze(ind_A[:, m].toarray())
            A[ind_pixels, m] = np.clip(
                A[ind_pixels, m] + ((U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) / V[m, m]), 0, np.inf)
        for m in range(nb):  # background
            A[:, K + m] = np.clip(A[:, K + m] + ((U[K + m] -
                                                  V[K + m].dot(A.T)) / V[K + m, K + m]), 0, np.inf)
    return A


def hals(Y, A, C, b, f, bSiz=3, maxIter=5):
    """ Hierarchical alternating least square method for solving NMF problem
    Y = A*C + b*f
    input:
    ------
       Y:      d1 X d2 [X d3] X T, raw data.
        It will be reshaped to (d1*d2[*d3]) X T in this
       function
       A:      (d1*d2[*d3]) X K, initial value of spatial components
       C:      K X T, initial value of temporal components
       b:      (d1*d2[*d3]) X nb, initial value of background spatial component
       f:      nb X T, initial value of background temporal component
       bSiz:   int or tuple of int
        blur size. A box kernel (bSiz X bSiz [X bSiz]) (if int) or bSiz (if tuple) will
        be convolved with each neuron's initial spatial component, then all nonzero
       pixels will be picked as pixels to be updated, and the rest will be
       forced to be 0.
       maxIter: maximum iteration of iterating HALS.
    output:
    -------
        the updated A, C, b, f
    @Author: Johannes Friedrich, Andrea Giovannucci
    See Also:
        http://proceedings.mlr.press/v39/kimura14.pdf
    """

    # smooth the components
    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    K = A.shape[1]  # number of neurons
    nb = b.shape[1]  # number of background components
    if isinstance(bSiz, (int, float)):
        bSiz = [bSiz] * len(dims)
    ind_A = ndi.uniform_filter(np.reshape(
        A, dims + (K,), order='F'), size=bSiz + [0])
    ind_A = np.reshape(ind_A > 1e-10, (np.prod(dims), K), order='F')
    ind_A = spr.csc_matrix(ind_A)  # indicator of nonnero pixels

    Ab = np.c_[A, b]
    Cf = np.r_[C, f.reshape(nb, -1)]
    for _ in range(maxIter):
        Cf = HALS4activity(np.reshape(Y, (np.prod(dims), T), order='F'),
                           Ab, Cf, K, nb)
        Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'),
                        Ab, Cf, K, nb, ind_A)

    return Ab[:, :-nb], Cf[:-nb], Ab[:, -nb:], Cf[-nb:].reshape(nb, -1)

######## Some general utils
