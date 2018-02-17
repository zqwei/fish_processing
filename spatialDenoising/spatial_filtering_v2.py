# from caiman.source_extraction.cnmf.pre_processing import get_noise_fft


# def spatial_filter_block(data,sn=None,maps=None,neuron_indx=None):
#     # spatial filter for small patches
#     dims = data.shape
#     mean_ = data.mean(2,keepdims=True)
#     data_ = data - mean_
#     sn = sn.reshape(np.prod(dims[:2]),order='F')
#     D = np.diag(sn)/sn.max() # data_std
#     data_r = data_.reshape((np.prod(dims[:2]),dims[2]),order='F')
#     data_r /= sn[:,np.newaxis]
#     Cy = data_r.dot(data_r.T)/(data_r.shape[1]-1)
#     try:
#         if neuron_indx is None:
#             hat_k = np.linalg.inv(Cy).dot(Cy-D)
#         else:
#             hat_k = np.linalg.inv(Cy).dot(Cy[neuron_indx,:]-D[neuron_indx,:])
#     except np.linalg.linalg.LinAlgError as err:
#         #if 'Singular matrix' in err.message:
#         print('Singular matrix bye bye')
#         # consider to be pseduo inverse instead
#         return data
#         #else:
#         #    raise
#     data_r *=sn[:,np.newaxis]
#     if neuron_indx is None:
#         y_ = hat_k.dot(data_r)
#     else:
#         y_ = data_r.copy()
#         y_[neuron_indx,:] = hat_k[:,np.newaxis].T.dot(data_r)
#     y_hat = y_.reshape(dims[:2]+(dims[2],),order='F')
#     y_hat = y_hat + mean_
#     # Cn_y = cm.local_correlations(data)
#     # Cn_yh = cm.local_correlations(y_hat)
#     return y_hat


# def spatial_filter_image(Y,gHalf=[2,2],sn=None):
#     # gHalf -- half size of the surrounding patch
#     # sn -- noise level for each pixel
#     mean_ = Y.mean(axis=2,keepdims=True)
#     if sn is not None:
#         print('sn given')
#     else:
#         # Estimate the noise level for each pixel by averaging the power spectral density.
#         sn,_ = get_noise_fft(Y - mean_,noise_method='logmexp')

#     # Computes the correlation image for the input dataset Y
#     # rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
#     Cnb = cm.local_correlations(Y, eight_neighbours=True, swap_dim=True)
#     maps = [Cnb.min(),Cnb.max()]
#     Y_new = Y.copy()
#     Y_tmp = Y.copy()
#     d = Y.shape
#     n_pixels = np.prod(d[:-1])

#     center = np.zeros((n_pixels,2)) #2D arrays
#     for k in np.arange(n_pixels):
#         # if k % 1e3==0:
#         #    print('first k pixels %d'%k)
#         ij = np.unravel_index(k,d[:2])

#         # this line replaces the for-loop
#         center[k, :] = ij
#         # for c, i in enumerate(ij):
#         #    center[k, c] = i

#         # get a squared patch around center pixel
#         # patch index
#         ijSig = [[np.maximum(ij[c] - gHalf[c], 0), np.minimum(ij[c] + gHalf[c] + 1, d[c])] for c in range(len(ij))]
#         # patch data
#         Y_curr = Y[[slice(*a) for a in ijSig]]
#         # patch noise
#         sn_curr = sn[[slice(*a) for a in ijSig]]
#         cc1 = ij[0]-ijSig[0][0]
#         cc2 = ij[1]-ijSig[1][0]
#         # get the center pixel location in the cropped patch
#         neuron_indx = np.ravel_multi_index((cc1,cc2),Y_curr.shape[:2],order='F')
#         Y_tmp[[slice(*a) for a in ijSig]] = spatial_filter_block(Y_curr,sn=sn_curr,maps=maps,neuron_indx=neuron_indx)
#         # only keep the center pixel to the update
#         Y_new[ij[0],ij[1],:] = Y_tmp[ij[0],ij[1],:]

#     return Y_new
