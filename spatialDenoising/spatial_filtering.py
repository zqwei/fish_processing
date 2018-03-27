# Implement spatial filter for each pixel

import numpy as np
import caiman as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import scipy as sp

def noise_estimator(Y,range_ff=[0.25,0.5],method='logmexp'):
    dims = Y.shape
    if len(dims)>2:
        V_hat = Y.reshape((np.prod(dims[:2]),dims[2]),order='F')
    else:
        V_hat = Y.copy()
    sns = []
    for i in range(V_hat.shape[0]):
        ff, Pxx = sp.signal.welch(V_hat[i,:],nperseg=min(256,dims[-1]))
        ind1 = ff > range_ff[0]
        ind2 = ff < range_ff[1]
        ind = np.logical_and(ind1, ind2)
        #Pls.append(Pxx)
        #ffs.append(ff)
        Pxx_ind = Pxx[ind]
        sn = {
            'mean': lambda Pxx_ind: np.sqrt(np.mean(np.divide(Pxx_ind, 2))),
            'median': lambda Pxx_ind: np.sqrt(np.median(np.divide(Pxx_ind, 2))),
            'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(np.divide(Pxx_ind, 2)))))
        }[method](Pxx_ind)
        sns.append(sn)
    sns = np.asarray(sns)
    if len(dims)>2:
        sns = sns.reshape(dims[:2],order='F')
    return sns

def covariance_matrix(Y):
    """
    Same as np.cov
    For a tutorial visit: https://people.duke.edu/~ccc14/sta-663/LinearAlgebraReview.html
    # Covariance matrix: outer product of the normalized
    # matrix where every variable has zero mean
    # divided by the number of degrees of freedom
    """
    num_rvs , num_obs = Y.shape
    w = Y - Y.mean(1)[:, np.newaxis]
    Cy = w.dot(w.T)/(num_obs - 1)
    return Cy


def spatial_filter_image(Y_new, gHalf=[2,2], sn=None):
    """
    Apply a wiener filter to image Y_new d1 x d2 x T
    """
    mean_ = Y_new.mean(axis=2,keepdims=True)
    if sn is None:
        sn = noise_estimator(Y_new - mean_, method='logmexp')
        if 0:
            plt.title('Noise level per pixel')
            plt.imshow(sn)
            plt.colorbar()
            plt.show()
    else:
        print('sn given')
    Cnb = cm.local_correlations(Y_new)
    maps = [Cnb.min(), Cnb.max()]

    Y_new2 = Y_new.copy()
    Y_new3 = np.zeros(Y_new.shape)#Y_new.copy()

    d = np.shape(Y_new)
    n_pixels = np.prod(d[:-1])

    center = np.zeros((n_pixels,2)) #2D arrays

    k_hats=[]
    for pixel in np.arange(n_pixels):
        if pixel % 1e3==0:
            print('first k pixels %d'%pixel)
        ij = np.unravel_index(pixel,d[:2])
        for c, i in enumerate(ij):
            center[pixel, c] = i
        # Get surrounding area
        ijSig = [[np.maximum(ij[c] - gHalf[c], 0), np.minimum(ij[c] + gHalf[c] + 1, d[c])]
                for c in range(len(ij))]

        Y_curr = np.array(Y_new[[slice(*a) for a in ijSig]].copy(),dtype=np.float32)
        sn_curr = np.array(sn[[slice(*a) for a in ijSig]].copy(),dtype=np.float32)
        cc1 = ij[0]-ijSig[0][0]
        cc2 = ij[1]-ijSig[1][0]
        neuron_indx = int(np.ravel_multi_index((cc1,cc2),Y_curr.shape[:2],order='F'))
        Y_out , k_hat = spatial_filter_block(Y_curr, sn=sn_curr,
                maps=maps, neuron_indx=neuron_indx)
        Y_new3[ij[0],ij[1],:] = Y_out[cc1,cc2,:]
        k_hats.append(k_hat)

    return Y_new3, k_hats


def spatial_filter_block(data,sn=None,maps=None,neuron_indx=None):
    """
    Apply wiener filter to block in data d1 x d2 x T
    """
    data = np.asarray(data)
    dims = data.shape
    mean_ = data.mean(2,keepdims=True)
    data_ = data - mean_

    if sn is None:
        sn, _ = cm.source_extraction.cnmf.pre_processing.get_noise_fft(data_,noise_method='mean')

    sn = sn.reshape(np.prod(dims[:2]),order='F')
    D = np.diag(sn**2)
    data_r = data_.reshape((np.prod(dims[:2]),dims[2]),order='F')
    Cy = covariance_matrix(data_r)

    Cy = Cy.copy()
    try:
        if neuron_indx is None:
            hat_k = np.linalg.inv(Cy).dot(Cy-D)
        else:
            hat_k = np.linalg.inv(Cy).dot(Cy[neuron_indx,:]-D[neuron_indx,:])
    except np.linalg.linalg.LinAlgError as err:
        print('Singular matrix(?) bye bye')
        return data , []
    if neuron_indx is None:
        y_ = hat_k.dot(data_r)
    else:
        y_ = data_r.copy()
        y_[neuron_indx,:] = hat_k[:,np.newaxis].T.dot(data_r)
    y_hat = y_.reshape(dims[:2]+(dims[2],),order='F')
    y_hat = y_hat + mean_

    Cn_y = cm.local_correlations(data)
    Cn_yh = cm.local_correlations(y_hat)

    # Plot the Cn for original and denoised image
    if False:
        fig,ax = plt.subplots(1,3,figsize=(10,5))
        im0 = ax[0].imshow(Cn_y.T,vmin=maps[0],vmax=maps[1])
        if neuron_indx is None:
            im1 = ax[1].imshow(hat_k)
        else:
            im1 = ax[1].imshow(hat_k[:,np.newaxis].T)
        im2 = ax[2].imshow(Cn_yh.T,vmin=maps[0],vmax=maps[1])
        ax[0].set_title('y')
        ax[1].set_title('k')
        ax[2].set_title('y_hat')

        ax[0].set_xticks(np.arange(y_hat.shape[0]))
        ax[0].set_yticks(np.arange(y_hat.shape[1]))
        ax[2].set_xticks(np.arange(y_hat.shape[0]))
        ax[2].set_yticks(np.arange(y_hat.shape[1]))
        ax[1].set_yticks(np.arange(1))
        if neuron_indx is None:
            ax[1].set_xticks(np.arange(np.prod(y_hat.shape[:2]))[::4])
            ax[1].set_yticks(np.arange(np.prod(y_hat.shape[:2]))[::4])

        divider0 = make_axes_locatable(ax[0])
        cax0 = divider0.append_axes("bottom", size="5%", pad=0.5)
        cbar0 = plt.colorbar(im0, cax=cax0, orientation='horizontal')
        divider1 = make_axes_locatable(ax[1])
        cax1 = divider1.append_axes("bottom", size="5%", pad=0.5)
        cbar1 = plt.colorbar(im1, cax=cax1, format="%.2f", orientation='horizontal')
        divider2 = make_axes_locatable(ax[2])
        cax2 = divider2.append_axes("bottom", size="5%", pad=0.5)
        cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
        plt.tight_layout()
        plt.show()
    return y_hat , hat_k
