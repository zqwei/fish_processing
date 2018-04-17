from __future__ import division, print_function
import numpy as np
import scipy.sparse as spr
import scipy
from . import trefide as trefide
from . import merging
import os
import sys
from scipy.ndimage import median_filter, maximum_filter
from .utils import gaussian_bandpass, HALS4activity, HALS4shape
from skimage import measure
from caiman.summary_images import local_correlations_fft
from caiman.source_extraction.cnmf.spatial import update_spatial_components
import matplotlib.pyplot as plt

"""
Organizational improvements:
[X] remove all unused params/vars
[X] propogate all new params/vars to top level funcs
[X] replace horrendous indexing
[X] renaming lettered data vars: Y, A, C, ect.
[X] Param to choose quantile & mean threshold options when intializing spatial footprints
[ ] modularize functions
    - [X] Remove Redundant Input Parsing Functions
    - [X] extract_components broken down into sublevels
    - [ ] break down algo into steps each having a separate func)
[X] remove horrendous reformatting (expect to only accept 2D videos)
[ ] old string docs updated
[ ] new string docs added
"""

"""
Potential technical improvements to evaluate:
[X] when thresholding connected components, take median over only POSITIVE elements
[X] Confirm all copying to use bare-minimum memory
[X] Estimate noise std instead of std of entire signal + noise
[ ] Update local corr neighborhood AROUND extracted patch
[ ] re-evaluate defaults to choose reasonable set that works across all/most q-state videos
[X] when highpass filtering data use other filters (wavelet, butterworth, ect.)
"""

"""
Technical modificataions inlcuded improve init quality over MATLAB implementation:
- Choose threshold over only POSITIVE elements when initializing spatial footprints
- Quantile & Mean threshold options when intializing spatial footprints
- Add additional filtering/updates after a component has been intialize
- Estimate stdv/pixel using raw data and pwelch/fft estimators
- Using different highpass filter
"""


def initialize_vimaging_raw(data_raw, **kwargs):
    """
    Called when want components returned relative to raw data
    """
    # Initialize spatial and temporal components against highpass data
    (spatial_components,
     temporal_components,
     centers) = initialize_vimaging_highpass(data_raw=data_raw, **kwargs)[:3]

    # Extract dimensions of the data
    num_frames, dims = data_raw.shape[0], data_raw.shape[1:]

    # Regress against raw data to obtain new temporal components
    if kwargs.get('verbose'):
        print("Refining temporal components against raw data...")
    temporal_components = HALS4activity(
        np.reshape(data_raw, (num_frames, np.prod(dims)), order='F').T,
        spatial_components.todense(),
        temporal_components,
        spatial_components.shape[-1],
        nb=0,
        iters=5
    )

    # Compute new background from residual with raw data
    residual = np.subtract(
        np.reshape(data_raw, (num_frames, np.prod(dims)), order='F').T,
        spatial_components.dot(temporal_components)
    )

    # Init background as top rank_bg components of the SVD(residual)
    spatial_bg, scale_bg, temporal_bg = spr.linalg.svds(
        residual,
        k=kwargs.get('rank_bg')
    )
    temporal_bg *= scale_bg[:, np.newaxis]

    return (spatial_components,
            temporal_components,
            centers.T,
            spatial_bg,
            temporal_bg)


def initialize_vimaging_rank2(data_raw, data_highpass, **kwargs):
    """
    Intializes both slow and fast components for each neuron
    """
    # Create LP Data For Use After Init
    data_lp = np.subtract(data_raw, data_highpass)

    # Initialize spatial and temporal components against highpass data
    (spatial_components_hp,
     temporal_components_hp,
     centers) = initialize_vimaging_highpass(data_highpass=data_highpass, **kwargs)[:3]

    # Extract dimensions of the data
    num_frames, dims = data_raw.shape[0], data_raw.shape[1:]

    # Regress against raw data to obtain new temporal components
    temporal_components_lp = np.array(
        [data_lp[:, center[1], center[0]] for center in centers]
    )
    temporal_components_lp = HALS4activity(
        np.reshape(data_lp, (num_frames, np.prod(dims)), order='F').T,
        spatial_components_hp.todense(),
        np.zeros(temporal_components_hp.shape),
        spatial_components_hp.shape[-1],
        nb=0,
        iters=5
    )

    spatial_components_lp = HALS4shape(
        np.reshape(data_lp, (num_frames, np.prod(dims)), order='F').T,
        np.array(spatial_components_hp.copy().todense()),
        temporal_components_lp,
        spatial_components_hp.shape[-1],
        nb=0,
        ind_A=(spatial_components_hp > 0),
        iters=5
    )

    # Compute new background from residual with raw data
    residual = np.subtract(
        np.reshape(data_raw, (num_frames, np.prod(dims)), order='F').T,
        scipy.sparse.hstack([spatial_components_hp, spatial_components_lp]).dot(
            np.vstack([temporal_components_hp, temporal_components_lp])
        )
    )

    # Init background as top rank_bg components of the SVD(residual)
    spatial_bg, scale_bg, temporal_bg = spr.linalg.svds(
        residual,
        k=kwargs.get('rank_bg')
    )
    temporal_bg *= scale_bg[:, np.newaxis]

    return (scipy.sparse.hstack([spatial_components_hp, spatial_components_lp]),
            np.vstack([temporal_components_hp, temporal_components_lp]),
            centers.T,
            spatial_bg,
            temporal_bg)


def initialize_vimaging_highpass(data_raw=None,
                                 data_highpass=None,
                                 stdv_pixel=None,
                                 default_kernel_sigma=2,
                                 default_kernel_len=25,
                                 max_neurons=30,
                                 patch_radius=5,
                                 rank_bg=1,
                                 min_corr=0.8,
                                 min_pnr=10,
                                 min_pixel=3,
                                 noise_thresh=3,
                                 boundary=None,
                                 visualize=False,
                                 verbose=True):
    """
    Initalize components for demixing voltage imaging data:

    This method uses a greedy approach of identifying ROIs from which to
    intialize neuron shapes and temporal components for NMF. ...
    uses highpass... options...

    Parameters:
    ----------
    data_raw: np.ndarray
        T x d1 x d2 movie, raw data.
    data_highpass: np.ndarray
        T x d1 x d2 movie, highpass filtered data.
    max_neurons: [optional] int
        number of neurons to extract (default value: 30). Maximal number for method 'corr_pnr'.
    min_corr: float
        minimum local correlation coefficients for selecting a seed pixel.
    min_pnr: float
        minimum peak-to-noise ratio for selecting a seed pixel.
    patch_radius: [optional] list,tuple
        size of kernel (default 5).
    maxIter: [optional] int
        number of iterations for HALS algorithm (default 5).
    use_hals: [optional] bool
        Whether to refine components with the hals method
    center_psf: Boolean
        True indicates centering the filtering kernel for background
        removal. This is useful for data with large background
        fluctuations.
    rank_bg: integer
        number of background components for approximating the background using NMF model
    stdv_pixel: ndarray
        per pixel noise
    rank_bg: integer
            number of background components for approximating the background using NMF model
    Returns:
    --------
    spatial_components: np.ndarray
        (d1*d2) x max_neurons , spatial filter of each neuron.
    temporal_components: np.ndarray
        T x max_neurons , calcium activity of each neuron.
    centers: np.ndarray
        max_neurons x 2 [or 3] , inferred centers of each neuron.
    bin: np.ndarray
        (d1*d2) x rank_bg, initialization of spatial background.
    fin: np.ndarray
        rank_bg x T matrix, initalization of temporal background
    Raise:
    ------
        Exception(
            "Either min_corr or min_pnr are None. Both of them must be real numbers.")
    """

    # --------------------------
    # -----Input Validation-----
    # --------------------------
    if verbose:
        print("Validating inputs...")

    # Require that some data be provided and shape of provided data match
    if data_raw is None and data_highpass is None:
        raise Exception(
            "Neither raw nor highpass filtered data has been provided."
        )

    if not (data_raw is None or data_highpass is None):
        if data_highpass.shape != data_raw.shape:
            raise Exception(
                "The shape of the provided data highpass and raw data do not match"
            )

    # --------------------------
    # -----Input Processing-----
    # --------------------------
    if verbose:
        print("Preprocessing inputs...")

    # Default boundary size depends on neuron bounding box size
    if boundary is None:
        boundary = int(round(patch_radius / 2))

    # Temporally highpass filter each pixel if raw data provided
    if data_highpass is None:
        if verbose:
            print("...Highpass filtering raw data...")
        data_highpass = gaussian_bandpass(data_raw,
                                          default_kernel_len,
                                          default_kernel_sigma,
                                          axis=0)
    else:
        data_highpass = data_highpass.copy()

    # Estimate pixel-wise noise standard deviations
    if stdv_pixel is None:
        if verbose:
            print("...Calculating stdv of each highpass pixel...")
        stdv_pixel = np.sqrt(np.var(data_highpass, axis=0))

    # --------------------------
    # ----ROI Identification----
    # --------------------------
    if verbose:
        print('Extracting Regions Of Interest...')

    # Extract highpass components
    spatial_components, temporal_components, centers = greedyROI_vimaging(
        data_highpass,
        stdv_pixel,
        max_neurons=max_neurons,
        patch_radius=patch_radius,
        min_corr=min_corr,
        min_pnr=min_pnr,
        min_pixel=min_pixel,
        boundary=boundary,
        noise_thresh=noise_thresh,
        verbose=verbose
    )

    # --------------------------
    # -----Background Init------
    # --------------------------
    if verbose:
        print('Estimating low rank background...')

    # Extract dimensions of the data
    num_frames, dims = data_highpass.shape[0], data_highpass.shape[1:]

    # Compute residual against initialized components
    residual = np.subtract(
        np.reshape(data_highpass, (num_frames, np.prod(dims)), order='F').T,
        spatial_components.dot(temporal_components)
    )

    # Init background as top rank_bg components of the SVD(residual)
    spatial_bg, scale_bg, temporal_bg = spr.linalg.svds(residual, k=rank_bg)
    temporal_bg *= scale_bg[:, np.newaxis]

    # --------------------------
    # ----Optional Plotting-----
    # --------------------------
    # if visualize:
    # spatial_tmp = np.reshape(np.array(spatial_components.todense()),
    #                          dims + (spatial_components.shape[-1],),
    #                          order='F')
    # spatial_summary(spatial_tmp,
    #                dims=dims)
    # plot_components(spatial_tmp,
    #                temporal_components,
    #                dims=dims)

    return (spatial_components,
            temporal_components,
            centers,
            spatial_bg.astype(np.float32),
            temporal_bg.astype(np.float32))


def greedyROI_vimaging(data_highpass, stdv_pixel, **kwargs):
    """
    using greedy method to initialize neurons by selecting pixels with large
    local correlation and large peak-to-noise ratio
    Args:
        data: np.ndarray (3D)
            the data used for initializing neurons. its dimension can be
            d1*d2*T.
        max_neurons: integer
            maximum number of neurons to be detected. If None, then the
            algorithm will stop when all pixels are below the thresholds.
        patch_radius: float
            average diameter of a neuron
        center_psf: Boolean
            True indicates centering the filtering kernel for background
            removal. This is useful for data with large background
            fluctuations.
        min_corr: float
            minimum local correlation coefficients for selecting a seed pixel.
        min_pnr: float
            minimum peak-to-noise ratio for selecting a seed pixel.
        min_pixel: integer
            minimum number of nonzero pixels for one neuron.
        boundary: integer
            pixels that are boundary pixels away from the boundary will
            be ignored for initializing neurons.
        noise_thresh: float
            pixel values smaller than noise_thresh*noise will be set as 0
            when computing the local correlation image.
        swap_dim: Boolean
            True indicates that time is listed in the last axis of data (matlab
            format)
    Returns:
        spatial_components: np.ndarray (d1*d2*T)
            spatial components of all neurons
        temporal_components: np.ndarray (K*T)
            nonnegative and denoised temporal components of all neurons
        centers: np.ndarray
            centers localtions of all neurons
    """
    # Unpack Relevant Kwargs
    patch_radius = kwargs.get('patch_radius')
    min_pnr = kwargs.get('min_pnr')
    min_corr = kwargs.get('min_corr')
    noise_thresh = kwargs.get('noise_thresh')
    boundary = kwargs.get('boundary')
    max_neurons = kwargs.get('max_neurons')

    # Extract data shape vars
    num_frames, dims = data_highpass.shape[0], data_highpass.shape[1:]

    # Use max value of each filtered pixel to compute peak to noise ratio
    data_max = np.max(data_highpass, axis=0)
    pnr_image = np.divide(data_max, stdv_pixel)
    pnr_image[pnr_image < min_pnr] = 0

    # Threshold highpass signal to separate noise from spiking activity
    data_spikes = np.copy(data_highpass) - \
        np.median(data_highpass, axis=0)[np.newaxis, :]
    data_spikes[data_spikes < noise_thresh * stdv_pixel] = 0

    # Generate local correlation image using only spiking activity
    corr_image = local_correlations_fft(data_spikes, swap_dim=False)

    # screen seed pixels as neuron centers
    v_search = corr_image * pnr_image
    v_search[(corr_image < min_corr) | (pnr_image < min_pnr)] = 0
    v_thresh = min_corr * min_pnr

    # Bool index matrix for the boundary pixels we ignore because of artifacts
    idx_boundary = np.ones(shape=dims).astype(np.bool)
    idx_boundary[boundary:-boundary, boundary:-boundary] = False

    # Whether or not a pixel has already been considered as seed
    idx_searched = np.logical_or(v_search <= 0, idx_boundary)

    # Set max_neurons default as a fraction of possible seed values
    if max_neurons is None:
        max_neurons = np.int32((idx_searched.size - idx_searched.sum()) / 5)

    # Pre-allocate space for intialized components
    spatial_components = np.zeros(
        shape=(max_neurons,) + dims, dtype=np.float32)
    temporal_components = np.zeros(shape=(max_neurons, num_frames),
                                   dtype=np.float32)
    centers = np.zeros(shape=(2, max_neurons))

    # Initialize loop vatriables
    num_neurons = 0
    continue_searching = True

    # Evaluate candidates until max_neuron inits or out of potential seeds
    while continue_searching:

        # Median filter seeding values to emphasize neuron bodies
        v_search = median_filter(v_search,
                                 size=(int(round(patch_radius / 4)),) * 2,
                                 mode='constant')
        #
        v_search[idx_searched] = 0
        idx_searched[v_search < v_thresh] = True
        # v_search[(corr_image < min_corr) | (pnr_image < min_pnr)] = 0  # add back in?

        # Max filter values so as to only choose local maximums as candidates
        v_max = maximum_filter(v_search,
                               size=(2 * int(round(patch_radius / 4) + 1),) * 2,
                               mode='constant')

        # Identify location of local maximums
        [row_loc_max, col_loc_max] = np.where(np.logical_and(v_search == v_max,
                                                             v_max > 0))
        local_maximums = v_max[row_loc_max, col_loc_max]

        # Get an ordering of the local maximums by descending corr * pnr value
        if not local_maximums.any():
            break  # no more candidates for seed pixels
        else:
            order_local_maximums = local_maximums.argsort()[::-1]

        # try to initialization neurons given all seed pixels
        for idx in order_local_maximums:
            row_idx = row_loc_max[idx]
            col_idx = col_loc_max[idx]

            # Make sure pixel will not be selected again in the future
            idx_searched[row_idx, col_idx] = True
            if v_search[row_idx, col_idx] < v_thresh:
                # skip this pixel if it's not suffitcent for being a seed pixel
                continue

            # crop a small box for estimation of components
            r_min = max(0, row_idx - patch_radius)
            r_max = min(dims[0], row_idx + patch_radius + 1)
            c_min = max(0, col_idx - patch_radius)
            c_max = min(dims[1], col_idx + patch_radius + 1)

            vid_slice = np.s_[:, r_min:r_max, c_min:c_max]
            img_slice = np.s_[r_min:r_max, c_min:c_max]
            patch_dims = (r_max - r_min, c_max - c_min)
            idx_ctr = np.ravel_multi_index((row_idx - r_min, col_idx - c_min),
                                           dims=patch_dims)

            data_highpass_box = data_highpass[vid_slice].reshape(
                -1, np.product(patch_dims))
            data_thresh_box = data_spikes[vid_slice].reshape(
                -1, np.product(patch_dims))

            # Extract temporal and spatial components from selected patch
            [spatial_component, temporal_component, success] = extract_components(
                data_thresh_box,
                data_highpass_box,
                idx_ctr,
                patch_dims,
                kwargs.get('min_pixel')
            )

            # Remove intialized component from data before continuing search
            if not success:
                continue  # bad initialization. discard and continue
            else:
                # Store Results
                centers[:, num_neurons] = [col_idx, row_idx]
                spatial_components[num_neurons][img_slice] = spatial_component
                temporal_components[num_neurons] = temporal_component

                # Update Filtered Data
                data_highpass[vid_slice] -= \
                    spatial_component[np.newaxis, ...] *\
                    temporal_component[..., np.newaxis, np.newaxis]

                # Update PNR Image
                data_max[img_slice] = np.max(data_highpass[vid_slice], axis=0)
                pnr_image[img_slice] = np.divide(data_max[img_slice],
                                                 stdv_pixel[img_slice])
                pnr_image[img_slice][pnr_image[img_slice] < min_pnr] = 0

                # Update Thresholded Data
                data_spikes[vid_slice][data_highpass[vid_slice]
                                       < noise_thresh * stdv_pixel[img_slice]] = 0

                # Naively Update Corr Image # TODO
                corr_image[img_slice] = local_correlations_fft(
                    data_spikes[vid_slice], swap_dim=False)
                v_search[img_slice] = corr_image[img_slice] * \
                    pnr_image[img_slice]
                v_search[np.logical_or(np.logical_or(idx_searched, idx_boundary),
                                       np.logical_or(corr_image < min_corr,
                                                     pnr_image < min_pnr))] = 0

                # Re-filter patch to prevent double intializing
                v_search[img_slice] = median_filter(v_search,  # TODO
                                                    size=(
                                                        int(round(patch_radius / 4)),) * 2,
                                                    mode='constant')[img_slice]

                # Increment counters, check continuation criterion, and report progress
                num_neurons += 1
                if num_neurons == max_neurons:
                    continue_searching = False
                    break
                else:
                    if num_neurons % 10 == 0 and kwargs.get('verbose'):
                        print('{:03d} neurons have been initialized'.format(
                            num_neurons))

    print('In total, ', num_neurons, 'neurons were initialized.')
    return (scipy.sparse.csc_matrix(np.reshape(
        spatial_components[:num_neurons],
        (-1, np.prod(dims)),
        order='F').transpose()
    ),
        temporal_components[:num_neurons],
        centers[:, :num_neurons].T.astype(int))


def extract_components(data_thresh,
                       data_highpass,
                       idx_ctr,
                       patch_dims,
                       min_pixel):
    """
    Extract temporal and spatial components from a patch of data centered around
    the seed pixel
    """

    # avoid empty results
    if np.sum(data_thresh[:, idx_ctr] > 0) < 1:
        return None, None, False

    # Extract first spatial component from thresholded spiking activity of center neuron
    spatial_component = _extract_spatial(data_thresh,
                                         data_thresh[:, idx_ctr],
                                         patch_dims,
                                         idx_ctr,
                                         thresh_method='mean')

    # avoid empty results
    if np.sum(spatial_component > 0) < min_pixel:
        return None, None, False

    # Refine temporal component by regressing spatial footprint against highpass filtered data
    temporal_component = np.dot(
        data_highpass, spatial_component) / np.dot(spatial_component, spatial_component)

    # Extract new spatial component from highpass filtered data using the updated temporal component
    spatial_component = _extract_spatial(data_highpass,
                                         temporal_component,
                                         patch_dims,
                                         idx_ctr,
                                         thresh_method='mean')

    # avoid empty results
    if np.sum(spatial_component > 0) < min_pixel:
        return None, None, False

    return (spatial_component.reshape(patch_dims),
            temporal_component.reshape(len(temporal_component)),
            True)


def _extract_spatial(data,
                     temporal_component,
                     patch_dims,
                     idx_ctr,
                     thresh_method='mean'):
    """
    During Initialization, regress current temporal component against data
    within bounding box to extract a matching spatial component
    """

    # Regress temporal component onto data to obtain initial spatial estimate
    spatial_component = np.dot(data.T, temporal_component) /\
        np.dot(temporal_component, temporal_component)

    # Enforce non-negativity constraint
    spatial_component[spatial_component < 0] = 0

    # Choose thresholding method used to enforce sparsity and connectedness
    if type(thresh_method) is float:
        quantile = thresh_method
        thresh_method = 'quantile'

    thresh_func = {
        'mean': lambda x: x > np.mean(x[x > 0]),
        'median': lambda x: x > np.median(x[x > 0]),
        'quantile': lambda x: x > np.percentile(x[x > 0], quantile * 100)
    }[thresh_method]

    # Enforce connectedness by only keeping thresholded pixel connected to seed
    conn_comp = measure.label(thresh_func(spatial_component).reshape(patch_dims),
                              connectivity=1).ravel()
    spatial_component[conn_comp != conn_comp[idx_ctr]] = 0

    return spatial_component


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def nmf_iter(data,
             temporal_components,
             temporal_bg,
             spatial_components,
             spatial_bg,
             sn,
             nb=2,
             n_iter=1,
             denoise_temporal=False,
             visualize=False,
             merge_after=None,
             disc_idx=None,
             buffer=10,
             mergingthr=0.5):
    """  """
    # Extract video dimensions
    d1, d2, T = data.shape

    # Artifact index default
    keep_idx = np.ones(T, dtype='bool')
    if disc_idx is not None:
        for idx in disc_idx:
            keep_idx[max(idx - buffer, 0):min(idx + buffer, T - 1)] = False

    K = np.inf
    while spatial_components.shape[-1] < K:

        # Loop Control
        K = spatial_components.shape[-1]

        # Perform n_iter updatesof both spatial & temporal components
        for itr in np.arange(n_iter):

            # Update spatial components using LARS
            with HiddenPrints():
                (spatial_components,
                 spatial_bg,
                 temporal_components,
                 temporal_bg) = update_spatial_components(
                     np.maximum(data.reshape(
                         d1 * d2, T, order='F'), 0)[:, keep_idx],
                     C=temporal_components[:, keep_idx],
                     f=temporal_bg[:, keep_idx],
                     A_in=spatial_components,
                     b_in=None,
                     sn=sn.reshape((d1 * d2,), order='F'),
                     dims=(d1, d2),
                     method='dilate',
                     nb=nb,
                     update_background_components=True
                )

            # TODO: Try HALS instead of matrix inverse?
            # Regress spatial components & bg against data to update temporal components & bg
            X = scipy.sparse.hstack((spatial_components, spatial_bg)).tocsc()
            temporal_components = np.array(np.linalg.inv(X.T.dot(X).todense()).dot(
                X.T.dot(data.reshape((d1 * d2, T), order='F'))))
            temporal_bg = temporal_components[spatial_components.shape[-1]:, :]
            temporal_components = temporal_components[:
                                                      spatial_components.shape[-1], :]

            # Apply Trend Filter To Smooth
            if denoise_temporal:
                temporal_components = trefide.denoise(temporal_components,
                                                      region_active_discount=.1)

            # apply non-negativity constraint
            temporal_components = np.maximum(temporal_components, 0)

            # Merge Components
            if merge_after and ((itr + 1) % merge_after == 0):
                (spatial_components,
                 temporal_components,
                 _, _) = merging.merge_components(spatial_components,
                                                  temporal_components, thr=mergingthr)
                if denoise_temporal:  # TODO: replace with option in merge
                    temporal_components = trefide.denoise(temporal_components,
                                                          region_active_discount=.1)
    # --------------------------
    # ----Optional Plotting-----
    # --------------------------
    # if visualize:
        # Reshape spatial components for plotting
        # spatial_tmp = np.reshape(np.array(spatial_components.todense()),
        #                          (d1, d2, spatial_components.shape[-1]),
        #                          order='F')

        # Display refined components
        # spatial_summary(spatial_tmp,
        #                dims=(d1, d2))
        # plot_components(spatial_tmp,
        #                temporal_components,
        #                dims=(d1, d2))

    return spatial_components, spatial_bg, temporal_components, temporal_bg


# --------------------------------
# ----Plotting TODO: RELOCATE-----
# --------------------------------

def spatial_summary(spatial_components, dims, n_col=5):
    K = spatial_components.shape[-1]

    # normalize footprints
    #spatial_components = spatial_components / (spatial_components ** 2).sum(0)

    # Create Image Of Cumulative Extracted Footprints
    overlay = np.sum(
        np.array(spatial_components.todense()), axis=-1).reshape(dims, order='F')

    # Generate
    n_row = int(np.ceil(K / n_col))
    plt.figure(figsize=(12, 2))
    plt.title('Cumulative Spatial Components')
    plt.imshow(overlay, cmap='nipy_spectral_r')
    # plt.axis('off')
    plt.tight_layout()
    plt.show()
    for row_idx in range(n_row):
        plt.figure(figsize=(12, 2))
        for col_idx in range(n_col):
            if ((row_idx * n_col) + col_idx) >= spatial_components.shape[-1]:
                pass
            else:
                idx = (row_idx * n_col) + col_idx
                plt.subplot(100 + (10 * n_col) + (col_idx + 1))
                footprint = np.reshape(np.array(spatial_components[:, idx].todense()),
                                       dims, order='F')
                patch_idx = np.argwhere(footprint > 0)
                # r1, r2 = np.min(patch_idx[:, 0]), np.max(patch_idx[:, 0])
                c1, c2 = np.min(patch_idx[:, 1]), np.max(patch_idx[:, 1])
                stretch = int(np.round((80 - (c2 - c1)) / 2))
                if stretch <= 0:
                    c1 = c1 - stretch
                    c2 = c2 + stretch
                elif c2 + stretch >= dims[1]:
                    c2 = dims[1]
                    c1 = c2 - 80
                elif c1 - stretch < 0:
                    c1 = 0
                    c2 = 80
                else:
                    c1 = c1 - stretch
                    c2 = c2 + stretch

                plt.imshow(footprint[:, c1:c2],
                           cmap='nipy_spectral_r')
                plt.title('SC #{}'.format(idx))

    return None


def plot_components(spatial_components, temporal_components, dims, trend_components=None,
                    indices=None, individually=True, noisey=None):
    d1, d2 = dims

    try:
        spatial_components = np.reshape(np.array(spatial_components.todense()),
                                        (d1, d2, spatial_components.shape[-1]),
                                        order='F')
    except:
        spatial_components = np.reshape(spatial_components,
                                        (d1, d2, spatial_components.shape[-1]),
                                        order='F')

    if indices is None:
        indices = np.arange(temporal_components.shape[0])

    if individually:
        for idx in indices:
            if trend_components is not None:
                signal = temporal_components[idx, :]
                plt.figure(figsize=(12, 8))
                plt.subplot(411)
                plt.title('Spiking Temporal Component {}'.format(idx))
                plt.plot(np.arange(len(signal)), signal)
                trend = trend_components[idx, :]
                plt.subplot(412)
                plt.title('Trend Temporal Component {}'.format(idx))
                plt.plot(np.arange(len(trend)), trend)
                plt.subplot(413)
                plt.title('Full Temporal Component {}'.format(idx))
                plt.plot(np.arange(len(trend)), signal + trend)
                plt.subplot(414)
                plt.title('Spatial Component {}'.format(idx))
                plt.imshow(
                    spatial_components[:, :, idx], cmap='nipy_spectral_r')
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            else:
                signal = temporal_components[idx, :]
                plt.figure(figsize=(16, 3))
                plt.subplot(121)
                plt.title('Spatial Component {}'.format(idx))
                patch_idx = np.argwhere(spatial_components[:, :, idx] > 0)
                # r1, r2 = np.min(patch_idx[:, 0]), np.max(patch_idx[:, 0])
                c1, c2 = np.min(patch_idx[:, 1]), np.max(patch_idx[:, 1])
                stretch = int(np.round((100 - (c2 - c1)) / 2))
                if stretch <= 0:
                    c1 = c1 - stretch
                    c2 = c2 + stretch
                elif c2 + stretch >= dims[1]:
                    c2 = dims[1]
                    c1 = c2 - 100
                elif c1 - stretch < 0:
                    c1 = 0
                    c2 = 100
                else:
                    c1 = c1 - stretch
                    c2 = c2 + stretch

                plt.imshow(spatial_components[:, c1:c2, idx],
                           cmap='nipy_spectral_r')
                # plt.imshow(
                #     spatial_components[:, :, idx], cmap='nipy_spectral_r')
                plt.axis('off')
                plt.subplot(122)
                plt.title('Temporal Component {}'.format(idx))
                if noisey is not None:
                    plt.plot(np.arange(len(signal)), noisey[idx, :], 'r')
                plt.plot(np.arange(len(signal)), signal, 'b')
                plt.tight_layout()
                plt.show()
    else:
        for group_indices in indices:
            fig, ax = plt.subplots(figsize=(18, 6))
            footprint = np.zeros([spatial_components.shape[0],
                                  spatial_components.shape[1],
                                  len(indices)])

            # Scaled Temporal Components
            for count, idx in enumerate(group_indices):
                signal = temporal_components[idx, :] - \
                    temporal_components[idx, : 400].mean()
                signal /= signal[1000: 2000].mean()
                ax.plot(np.arange(len(signal)), signal)
                footprint[:, :, count] = (
                    (spatial_components[:, :, idx] > 0) * (count + 1))
                ax.set(xlabel='time', ylabel='Fluorescence',
                       title='Temporal Components {}'.format(indices))
                ax.grid()
                ax.legend(indices)
                plt.show()

                # Overlapped Spatial Image
                plt.figure(figsize=(18, 9))
                plt.title('Spatial Components {}'.format(group_indices))
                plt.imshow(np.sum(footprint, axis=-1),
                           cmap='nipy_spectral_r')
                plt.axis('off')
                plt.show()
