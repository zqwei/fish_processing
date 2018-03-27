from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filt


def plot_components(spatial_components,
                    spike_components,
                    trend_components,
                    photo_components,
                    stim,
                    plot_idx,
                    dims,
                    zoom_idx=None,
                    noisey_components=None,
                    nn_estimate=None,
                    truth=None):
    d1, d2 = dims
    K, T = spike_components.shape

    # Find Transition to mark with vertical lines
    trans_idx = np.argwhere(~plot_idx)

    # compute spatial norms
    scales = np.linalg.norm(spatial_components.todense(),
                            axis=0)[:, np.newaxis]

    # Normalize Sptial Units
    # Move norms to temporal components
    spike_components = np.multiply(spike_components, scales)
    trend_components = np.multiply(trend_components, scales)
    photo_components = np.multiply(photo_components, scales)

    # Reshape Spatial For Plotting
    spatial_components = np.reshape(np.array(spatial_components.todense()),
                                    (d1, d2, K), order='F')

    for idx in np.arange(K):
        fix, ax = plt.subplots(4, 1, sharex=True, figsize=(13, 8))

        # Spiking Components
        ax[0].set_title('Temporal Component Neuron {}'.format(idx + 1))
        if noisey_components is not None:
            ax[0].plot(np.arange(T)[plot_idx] / 1000,
                       noisey_components[idx, plot_idx] * scales[idx], 'r')
        if nn_estimate is not None:
            ax[0].plot(np.arange(T)[plot_idx] / 1000,
                       truth[idx, plot_idx] * scales[idx], 'm')
            ax[0].plot(np.arange(T)[plot_idx] / 1000,
                       spike_components[idx, plot_idx], 'b')
            ax[0].plot(np.arange(T)[plot_idx] / 1000,
                       nn_estimate[idx, plot_idx] * scales[idx], 'g')
        else:
            ax[0].plot(np.arange(T)[plot_idx] / 1000,
                       spike_components[idx, plot_idx], 'b')
        if nn_estimate is not None:
            ax[0].legend(['Input', 'Target', 'TF', 'NN'])
        elif noisey_components is not None:
            ax[0].legend(['Raw', 'Trend Filtered'])

        for trans in trans_idx:
            ax[0].axvline(x=trans / 1000, c='k', ls='--')
        ax[0].set_ylabel('a.u.')
        ax[0].ticklabel_format(style='sci', scilimits=(0, 0))

        # Stim Trend
        ax[1].set_title('Slow Trend Neuron {}'.format(idx + 1))
        ax[1].plot(np.arange(T)[plot_idx] / 1000,
                   trend_components[idx, plot_idx], 'b')
        for trans in trans_idx:
            ax[1].axvline(x=trans / 1000, c='k', ls='--')
        ax[1].set_ylabel('a.u.')
        ax[1].ticklabel_format(style='sci', scilimits=(0, 0))

        # Photo trend
        ax[2].set_title('Photobleaching Decay Neuron {}'.format(idx + 1))
        ax[2].plot(np.arange(T)[plot_idx] / 1000,
                   photo_components[idx, plot_idx] - photo_components[idx, plot_idx].min(), 'b')
        for trans in trans_idx:
            ax[2].axvline(x=trans / 1000, c='k', ls='--')
        ax[2].set_ylabel('a.u.')
        ax[2].ticklabel_format(style='sci', scilimits=(0, 0))

        # Stimuli
        ax[3].set_title('Stimuli Protocol')
        ax[3].plot(np.arange(T)[plot_idx] / 1000,
                   stim[plot_idx], 'b')
        for trans in trans_idx:
            ax[3].axvline(x=trans / 1000, c='k', ls='--')

        plt.yticks([])
        plt.tight_layout()
        plt.xlabel('Time (s)')
        plt.show()

        if zoom_idx is not None:
            fix, ax = plt.subplots(1, 1, sharex=True, figsize=(2.5, 8))
            # Spiking Components
            if noisey_components is not None:
                ax.plot(np.arange(T)[plot_idx][zoom_idx] / 1000,
                        noisey_components[idx, plot_idx][zoom_idx] * scales[idx], 'r')
            if nn_estimate is not None:
                ax.plot(np.arange(T)[plot_idx][zoom_idx] / 1000,
                        truth[idx, plot_idx][zoom_idx] * scales[idx], 'm')
                ax.plot(np.arange(T)[plot_idx][zoom_idx] / 1000,
                        nn_estimate[idx, plot_idx][zoom_idx] * scales[idx], 'g')

            ax.plot(np.arange(T)[plot_idx][zoom_idx] / 1000,
                    spike_components[idx, plot_idx][zoom_idx], 'b')
            ax.set_ylabel('a.u.')
            ax.ticklabel_format(style='sci', scilimits=(0, 0))
            plt.tight_layout()
            plt.xlabel('Time (s)')
            plt.show()


def spatial_summary(spatial_components, dims, n_col=5,
                    label_shift=None,
                    figsize=(10, 8),
                    fontsize=18, transpose=False, individually=False):

    d1, d2 = dims
    K = spatial_components.shape[-1]

    # compute spatial norms
    scales = np.linalg.norm(spatial_components.todense(),
                            axis=0)[np.newaxis, :]

    # get location of labels
    inds = np.argmax(spatial_components.todense(), axis=0)
    cols, rows = np.unravel_index(inds, (d1, d2), order='F')
    rows = rows.squeeze()
    cols = cols.squeeze()
    if label_shift is not None:
        for idx, coord in enumerate(label_shift):
            rows[idx] += coord[0]
            cols[idx] += coord[1]

    # normalize spatial units
    spatial_components = np.reshape(np.divide(np.array(
        spatial_components.todense()), scales), dims + (K,), order='F')

    # Create Image Of Cumulative Extracted Footprints
    overlay = np.sum(spatial_components, axis=-1)
    plt.figure(figsize=figsize)
    plt.title('Cumulative Spatial Components', fontsize=2 * fontsize)
    plt.imshow(overlay, cmap='nipy_spectral_r')
    for idx, (row, col) in enumerate(zip(rows, cols)):
        plt.text(row, col, idx + 1, fontdict={'size': fontsize})
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Individual Plots
    if individually:
        for idx, (row, col) in enumerate(zip(rows, cols)):
            plt.figure(figsize=figsize)
            footprint = spatial_components[:, :, idx]
            plt.imshow(footprint, cmap='nipy_spectral_r')
            plt.text(row, col, idx + 1, fontdict={'size': fontsize * 1.5})
            plt.title('Spatial Component Neuron {}'.format(
                idx + 1), fontsize=fontsize * 1.5)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
