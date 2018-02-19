import trefide as tfd
import preprocess_blockSVD as pre_svd

def trial_one(Y_):
    def process_patch(data_in):
        dims = data_in.shape
        data_all = data_in.reshape((dims[0]*dims[1],dims[2]), order='F').T
        # in a 2d matrix, we get rid of any broke (nan) pixels
        # we assume fixed broken across time
        broken_idx = np.isinf(data_all[0,:])
        data = data_all[:, ~broken_idx]
        # Remove the data mean
        mu = data.mean(0, keepdims=True)
        data0 = data - mu
        U, s, Vt = pre_svd.compute_svd(data0.T, method='vanilla')
        ctid = pre_svd.choose_rank(Vt, maxlag=maxlag,iterate=iterate, confidence=confidence,corr=corr,kurto=kurto)
        n, L = Vt.shape
        vtid = np.zeros(shape=(3, n)) * np.nan
        mean_th = pre_svd.covCI_wnoise(L,confidence=confidence,maxlag=maxlag)
        #keep1 = pre_svd.cov_one(Vt, mean_th, maxlag=maxlag, iterate=iterate)
        ht_data=np.random.randn(L)
        covdata = pre_svd.axcov(ht_data, maxlag)[maxlag:]/ht_data.var()
        keep1 = np.where(np.logical_or(ctid[0, :] == 1, ctid[1, :] == 1))[0]
        #pre_svd.plot_vt_cov(Vt,keep1,maxlag)
        S = np.eye(len(keep1))*s[keep1.astype(int)]
        Yd_patch = U[:,keep1].dot(S.dot(Vt[keep1,:])) + mu.T
        Yd_patch =Yd_patch.reshape((dims[0],dims[1])+(dims[2],),order='F')
        Vt_hat = tfd.denoise(Vt[keep1,:],noise_estimator='fft',noise_method='logmexp')
        Yd_patch_filt = U[:,keep1].dot(S.dot(Vt_hat)) + mu.T
        Yd_patch_filt = Yd_patch_filt.reshape((dims[0],dims[1])+(dims[2],),order='F')
        return S.dot(Vt[keep1,:]),S.dot(Vt_hat)
    patches = pre_svd.split_image_into_blocks(Y_,k)
    #return process_patch(data_in)
    dimsMc = list(map(np.shape,patches))
    Vt1,Vt1d=[],[]
    shapes=[]
    for patch in patches:
        V1,V1d= process_patch(patch)
        Vt1.append(V1)
        Vt1d.append(V1d)
        shapes.append(V1.shape[0])
    return Vt1,Vt1d,shapes
#patches=pre_svd.split_image_into_blocks(Y_detr,k)
#data_in = patches[6]
