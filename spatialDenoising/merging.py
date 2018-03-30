# -*- coding: utf-8 -*-
"""Merging of spatially overlapping components that are temporally correlated
Created on Tue Sep  8 16:23:57 2015
@author: agiovann
"""
#\package caiman/source_extraction/cnmf
#\version   1.0
#\copyright GNU General Public License v2.0

from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
from scipy.sparse import coo_matrix, csgraph, csc_matrix, lil_matrix
import scipy
import numpy as np


def merge_components(A,
                     C,
                     dview=None,
                     thr=0.85,
                     fast_merge=True,
                     mx=1000):
    """ Merging of spatially overlapping components that have highly correlated temporal activity
    The correlation threshold for merging overlapping components is user specified in thr
Parameters:
-----------     
A: sparse matrix
     matrix of spatial components (d x K)
C: np.ndarray
     matrix of temporal components (K x T)
thr:   scalar between 0 and 1 
     correlation threshold for merging (default 0.85)
mx:    int
     maximum number of merging operations (default 50)
fast_merge: bool
    if true perform rank 1 merging, otherwise takes best neuron
Returns:
--------
A:     sparse matrix
        matrix of merged spatial components (d x K)
C:     np.ndarray
        matrix of merged temporal components (K x T)
nr:    int
    number of components after merging
merged_ROIs: list
    index of components that have been merged     
    """

    # only place Y is used ...TODO replace with A,C
    d, nr = A.shape
    t = C.shape[-1]

    # % find graph of overlapping spatial components
    A_corr = scipy.sparse.triu(A.T.dot(A))
    A_corr.setdiag(0)
    A_corr = A_corr.tocsc()
    FF2 = A_corr > 0
    C_corr = scipy.sparse.csc_matrix(A_corr.shape)
    for ii in range(nr):
        overlap_indeces = A_corr[ii, :].nonzero()[1]
        if len(overlap_indeces) > 0:
            # we chesk the correlation of the calcium traces for eahc overlapping components
            corr_values = [scipy.stats.pearsonr(C[ii, :], C[jj, :])[
                0] for jj in overlap_indeces]
            C_corr[ii, overlap_indeces] = corr_values

    FF1 = (C_corr + C_corr.T) > thr
    FF3 = FF1.multiply(FF2)

    nb, connected_comp = csgraph.connected_components(
        FF3)  # % extract connected components

    # p = temporal_params['p']
    list_conxcomp = []
    for i in range(nb):  # we list them
        if np.sum(connected_comp == i) > 1:
            list_conxcomp.append((connected_comp == i).T)
    list_conxcomp = np.asarray(list_conxcomp).T

    if list_conxcomp.ndim > 1:
        cor = np.zeros((np.shape(list_conxcomp)[1], 1))
        for i in range(np.size(cor)):
            fm = np.where(list_conxcomp[:, i])[0]
            for j1 in range(np.size(fm)):
                for j2 in range(j1 + 1, np.size(fm)):
                    cor[i] = cor[i] + C_corr[fm[j1], fm[j2]]

        if np.size(cor) > 1:
            # we get the size (indeces)
            ind = np.argsort(np.squeeze(cor))[::-1]
        else:
            ind = [0]

        nbmrg = min((np.size(ind), mx))   # number of merging operations

        # we initialize the values
        A_merged = lil_matrix((d, nbmrg))
        C_merged = np.zeros((nbmrg, t))
        merged_ROIs = []

        for i in range(nbmrg):
            merged_ROI = np.where(list_conxcomp[:, ind[i]])[0]
            merged_ROIs.append(merged_ROI)

            # we l2 the traces to have normalization values
            C_to_norm = np.sqrt([computedC.dot(computedC)
                                 for computedC in C[merged_ROI]])

            # from here we are computing initial values for C and A
            Acsc = A.tocsc()[:, merged_ROI]
            Ctmp = np.array(C)[merged_ROI, :]
            print((merged_ROI.T))

            # this is a  big normalization value that for every one of the merged neuron
            C_to_norm = np.sqrt(np.ravel(Acsc.power(2).sum(
                axis=0)) * np.sum(Ctmp ** 2, axis=1))
            indx = np.argmax(C_to_norm)

            if fast_merge:
                # we normalize the values of different A's to be able to compare them efficiently. we then sum them
                computedA = Acsc.dot(scipy.sparse.diags(
                    C_to_norm, 0, (len(C_to_norm), len(C_to_norm)))).sum(axis=1)

                # we operate a rank one NMF, refining it multiple times (see cnmf demos )
                for _ in range(10):
                    computedC = np.maximum(Acsc.T.dot(computedA).T.dot(
                        Ctmp) / (computedA.T * computedA), 0)
                    # computedC = Acsc.T.dot(computedA).T.dot(
                    #     Ctmp) / (computedA.T * computedA)
                    computedA = np.maximum(
                        Acsc.dot(Ctmp.dot(computedC.T)) / (computedC * computedC.T), 0)
            else:
                print('Simple Merging Take Best Neuron')
                computedC = Ctmp[indx]
                computedA = Acsc[:, indx]

            # then we de-normalize them using A_to_norm
            A_to_norm = np.sqrt(computedA.T.dot(computedA)[
                                0, 0] / Acsc.power(2).sum(0).max())
            computedA /= A_to_norm
            computedC *= A_to_norm

            # TODO: Add option to denoise temporal component via trefide?

            A_merged[:, i] = computedA
            C_merged[i, :] = computedC

        # we want to remove merged neuron from the initial part and replace them with merged ones
        neur_id = np.unique(np.hstack(merged_ROIs))
        good_neurons = np.setdiff1d(list(range(nr)), neur_id)
        A = scipy.sparse.hstack((A.tocsc()[:, good_neurons], A_merged.tocsc()))
        C = np.vstack((C[good_neurons, :], C_merged))
        nr = nr - len(neur_id) + nbmrg

    else:
        print('No neurons merged!')
        merged_ROIs = []

    return A, C, nr, merged_ROIs
