# ZW -- indicated the comments by Ziqiang Wei
# email : weiz@janelia.hhmi.org
import numpy as np
import time
import networkx as nx
from ..utils.memory import clear_variables
from sklearn.decomposition import NMF, TruncatedSVD
from scipy.sparse import csr_matrix, issparse, triu, csc_matrix
from scipy.stats import rankdata
import scipy.ndimage


def threshold_data(Yd, th=2):
    Yd_median = np.median(Yd, axis=-1, keepdims=True)
    Yd_mad = np.median(abs(Yd - Yd_median), axis=-1, keepdims=True)
    Yt = Yd-(Yd_median + th*Yd_mad)
    Yt[Yt<0]=0
    return Yt


def find_superpixel(Yt, cut_off_point, length_cut):
    dims = Yt.shape;
    ref_mat = np.arange(np.prod(dims[:-1])).reshape(dims[:-1],order='F')
    ######### calculate correlation ############
    # ZW -- better optimizaiton of memory here
    w_mov = (Yt.transpose(2,0,1) - np.mean(Yt, axis=2)) / np.std(Yt, axis=2);
    w_mov[np.isnan(w_mov)] = 0;
    # ZW -- this one need to be speed up ----
    rho_v = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_h = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)
    rho_l = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:]), axis=0)
    rho_r = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:]), axis=0)
    # clean for memory leak
    w_mov = None
    clear_variables(w_mov)
    # get correlation mats
    rho_v = np.concatenate([rho_v, np.zeros([1, rho_v.shape[1]])], axis=0)
    rho_h = np.concatenate([rho_h, np.zeros([rho_h.shape[0],1])], axis=1)
    rho_r = np.concatenate([rho_r, np.zeros([rho_r.shape[0],1])], axis=1)
    rho_r = np.concatenate([rho_r, np.zeros([1, rho_r.shape[1]])], axis=0)
    rho_l = np.concatenate([np.zeros([rho_l.shape[0],1]), rho_l], axis=1)
    rho_l = np.concatenate([rho_l, np.zeros([1, rho_l.shape[1]])], axis=0)
    ################## find pairs where correlation above threshold
    temp_v = np.where(rho_v > cut_off_point);
    A_v = ref_mat[temp_v];
    B_v = ref_mat[(temp_v[0] + 1, temp_v[1])]
    temp_h = np.where(rho_h > cut_off_point);
    A_h = ref_mat[temp_h];
    B_h = ref_mat[(temp_h[0], temp_h[1] + 1)]
    temp_l = np.where(rho_l > cut_off_point);
    A_l = ref_mat[temp_l];
    B_l = ref_mat[(temp_l[0] + 1, temp_l[1] - 1)]
    temp_r = np.where(rho_r > cut_off_point);
    A_r = ref_mat[temp_r];
    B_r = ref_mat[(temp_r[0] + 1, temp_r[1] + 1)]
    A = np.concatenate([A_v,A_h,A_l,A_r])
    B = np.concatenate([B_v,B_h,B_l,B_r])
    ########### form connected componnents #########
    G = nx.Graph();
    G.add_edges_from(list(zip(A, B)))
    comps=list(nx.connected_components(G))
    connect_mat=np.zeros(np.prod(dims[:2]));
    idx=0;
    for comp in comps:
        if(len(comp) > length_cut):
            idx = idx+1;
    permute_col = np.random.permutation(idx)+1;
    ii=0;
    for comp in comps:
        if(len(comp) > length_cut):
            connect_mat[list(comp)] = permute_col[ii];
            ii = ii+1;
    connect_mat_1 = connect_mat.reshape(dims[0],dims[1],order='F');
    return connect_mat_1, idx, comps, permute_col


def find_superpixel_3d(Yt, num_plane, cut_off_point, length_cut):
    dims = Yt.shape;
    Yt_3d = Yt.reshape(dims[0],int(dims[1]/num_plane),num_plane,dims[2],order="F");
    dims = Yt_3d.shape;
    ref_mat = np.arange(np.prod(dims[:-1])).reshape(dims[:-1],order='F');
    ######### calculate correlation ############
    w_mov = (Yt_3d.transpose(3,0,1,2) - np.mean(Yt_3d, axis=3)) / np.std(Yt_3d, axis=3);
    w_mov[np.isnan(w_mov)] = 0;
    rho_v = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_h = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)
    rho_l = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:]), axis=0)
    rho_r = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:]), axis=0)
    rho_u = np.mean(np.multiply(w_mov[:, :, :, :-1], w_mov[:, :, :, 1:]), axis=0)
    w_mov = None
    clear_variables(w_mov)
    # get correlation mats
    rho_v = np.concatenate([rho_v, np.zeros([1, rho_v.shape[1],num_plane])], axis=0)
    rho_h = np.concatenate([rho_h, np.zeros([rho_h.shape[0],1,num_plane])], axis=1)
    rho_r = np.concatenate([rho_r, np.zeros([rho_r.shape[0],1,num_plane])], axis=1)
    rho_r = np.concatenate([rho_r, np.zeros([1, rho_r.shape[1],num_plane])], axis=0)
    rho_l = np.concatenate([np.zeros([rho_l.shape[0],1,num_plane]), rho_l], axis=1)
    rho_l = np.concatenate([rho_l, np.zeros([1, rho_l.shape[1],num_plane])], axis=0)
    rho_u = np.concatenate([rho_u, np.zeros([rho_u.shape[0], rho_u.shape[1],1])], axis=2)
    ################## find pairs where correlation above threshold
    temp_v = np.where(rho_v > cut_off_point);
    A_v = ref_mat[temp_v];
    B_v = ref_mat[(temp_v[0] + 1, temp_v[1], temp_v[2])]
    temp_h = np.where(rho_h > cut_off_point);
    A_h = ref_mat[temp_h];
    B_h = ref_mat[(temp_h[0], temp_h[1] + 1, temp_h[2])]
    temp_u = np.where(rho_u > cut_off_point);
    A_u = ref_mat[temp_u];
    B_u = ref_mat[(temp_u[0], temp_u[1], temp_u[2]+1)]
    temp_l = np.where(rho_l > cut_off_point);
    A_l = ref_mat[temp_l];
    B_l = ref_mat[(temp_l[0] + 1, temp_l[1] - 1, temp_l[2])]
    temp_r = np.where(rho_r > cut_off_point);
    A_r = ref_mat[temp_r];
    B_r = ref_mat[(temp_r[0] + 1, temp_r[1] + 1, temp_r[2])]
    A = np.concatenate([A_v,A_h,A_l,A_r,A_u])
    B = np.concatenate([B_v,B_h,B_l,B_r,B_u])
    ########### form connected componnents #########
    G = nx.Graph()
    G.add_edges_from(list(zip(A, B)))
    comps=list(nx.connected_components(G))
    connect_mat=np.zeros(np.prod(dims[:-1]));
    idx=0;
    for comp in comps:
        if(len(comp) > length_cut):
            idx = idx+1;
    permute_col = np.random.permutation(idx)+1;
    ii=0;
    for comp in comps:
        if(len(comp) > length_cut):
            connect_mat[list(comp)] = permute_col[ii];
            ii = ii+1;
    connect_mat_1 = connect_mat.reshape(Yt.shape[:-1],order='F');
    return connect_mat_1, idx, comps, permute_col


def NMF_comps(c_, length_cut=None, Yt_r=None, model=None):
    u_, v_ = (None, None)
    if len(c_)>length_cut:
        dims = Yt_r.shape[0]
        y_temp = Yt_r[list(c_),:].astype('float')
        u_ = np.zeros((dims,1)).astype('float32')
        u_[list(c_)] = model.fit_transform(y_temp, W=np.array(y_temp.mean(axis=1)),H = np.array(y_temp.mean(axis=0)))
        v_ = model.components_.T
    return u_, v_,


def spatial_temporal_ini(Yt, comps, idx, length_cut, bg=False):
    """
    Apply rank-1-NMF to find spatial and temporal initialization for each superpixel in Yt.
    """
    dims = Yt.shape;
    T = dims[-1];
    Yt_r = Yt.reshape(np.prod(dims[:-1]),T,order = "F");
    Yt_r = csr_matrix(Yt_r);
    model = NMF(n_components=1, init='custom')
    U_mat = []
    V_mat = []
    for ii, comp in enumerate(comps):
        if(len(comp) > length_cut):
            y_temp = Yt_r[list(comp),:].astype('float')
            u_ = np.zeros((np.prod(dims[:-1]),1)).astype('float32')
            u_[list(comp)] = model.fit_transform(y_temp, W=np.array(y_temp.mean(axis=1)),H = np.array(y_temp.mean(axis=0)))
            U_mat.append(u_)
            V_mat.append(model.components_.T)
    if len(U_mat)>1:
        U_mat = np.concatenate(U_mat, axis=1)
        V_mat = np.concatenate(V_mat, axis=1)
    else:
        U_mat = np.zeros([np.prod(dims[:-1]),1]).astype('float32')
        V_mat = np.zeros([T,1]).astype('float32')
    if bg:
        bg_comp_pos = np.where(U_mat.sum(axis=1) == 0)[0]
        y_temp = Yt_r[bg_comp_pos,:]
        bg_u = np.zeros([Yt_r.shape[0],bg])
        y_temp = y_temp - y_temp.mean(axis=1)
        svd = TruncatedSVD(n_components=bg, n_iter=7, random_state=0)
        bg_u[bg_comp_pos,:] = svd.fit_transform(y_temp)
        bg_v = svd.components_.T
        bg_v = bg_v - bg_v.mean(axis=0)
    else:
        bg_v = None
        bg_u = None
    return V_mat, U_mat, bg_v, bg_u


def search_superpixel_in_range(connect_mat, permute_col, V_mat):
    unique_pix = np.asarray(np.sort(np.unique(connect_mat)),dtype="int");
    unique_pix = unique_pix[np.nonzero(unique_pix)];
    M = np.zeros([V_mat.shape[0], len(unique_pix)]);
    for ii in range(len(unique_pix)):
        M[:,ii] =  V_mat[:,int(np.where(permute_col==unique_pix[ii])[0])];
    return unique_pix, M


def fast_sep_nmf(M, r, th, normalize=1):
    """
    Find pure superpixels. solve nmf problem M = M(:,K)H, K is a subset of M's columns.

    Parameters:
    ----------------
    M: 2d np.array, dimension T x idx
        temporal components of superpixels.
    r: int scalar
        maximum number of pure superpixels you want to find.  Usually it's set to idx, which is number of superpixels.
    th: double scalar, correlation threshold
        Won't pick up two pure superpixels, which have correlation higher than th.
    normalize: Boolean.
        Normalize L1 norm of each column to 1 if True.  Default is True.

    Return:
    ----------------
    pure_pixels: 1d np.darray, dimension d x 1. (d is number of pure superpixels)
        pure superpixels for these superpixels, actually column indices of M.
    """

    pure_pixels = [];
    if normalize == 1:
        M = M/np.sum(M, axis=0,keepdims=True);
    normM = np.sum(M**2, axis=0,keepdims=True);
    normM_orig = normM.copy();
    normM_sqrt = np.sqrt(normM);
    nM = np.sqrt(normM);
    ii = 0;
    U = np.zeros([M.shape[0], r]);
    while ii < r and (normM_sqrt/nM).max() > th:
        temp = normM/normM_orig;
        pos = np.where(temp == temp.max())[1][0];
        pos_ties = np.where((temp.max() - temp)/temp.max() <= 1e-6)[1];
        if len(pos_ties) > 1:
            pos = pos_ties[np.where(normM_orig[0,pos_ties] == (normM_orig[0,pos_ties]).max())[0][0]];
        pure_pixels.append(pos);
        U[:,ii] = M[:,pos].copy();
        for jj in range(ii):
            U[:,ii] = U[:,ii] - U[:,jj]*sum(U[:,jj]*U[:,ii])
        U[:,ii] = U[:,ii]/np.sqrt(sum(U[:,ii]**2));
        normM = np.maximum(0, normM - np.matmul(U[:,[ii]].T, M)**2);
        normM_sqrt = np.sqrt(normM);
        ii = ii+1;
    pure_pixels = np.array(pure_pixels);
    return pure_pixels


def prepare_iteration(Yd, connect_mat_1, permute_col, pure_pix, U_mat, V_mat, more=False):
    """
    Get some needed variables for the successive nmf iterations.
    Parameters:
    ----------------
    Yt: 3d np.darray, dimension d1 x d2 x T
        thresholded data
    connect_mat_1: 2d np.darray, d1 x d2
        illustrate position of each superpixel, same value means same superpixel
    permute_col: list, length = number of superpixels
        random number used to idicate superpixels in connect_mat_1
    pure_pix: 1d np.darray, dimension d x 1. (d is number of pure superpixels)
        pure superpixels for these superpixels, actually column indices of M.
    V_mat: 2d np.darray, dimension T x number of superpixel
        temporal initilization
    U_mat: 2d np.darray, dimension (d1*d2) x number of superpixel
        spatial initilization

    Return:
    ----------------
    U_mat: 2d np.darray, number pixels x number of pure superpixels
        initialization of spatial components
    V_mat: 2d np.darray, T x number of pure superpixels
        initialization of temporal components
    brightness_rank: 2d np.darray, dimension d x 1
        brightness rank for pure superpixels in this patch. Rank 1 means the brightest.
    B_mat: 2d np.darray
        initialization of constant background
    normalize_factor: std of Y
    """
    dims = Yd.shape;
    T = dims[-1];
    Yd = Yd.reshape(np.prod(dims[:-1]),-1, order="F");
    ####################### pull out all the pure superpixels ################################
    permute_col = list(permute_col);
    pos = [permute_col.index(x) for x in pure_pix];
    U_mat = U_mat[:,pos];
    V_mat = V_mat[:,pos];
    ####################### order pure superpixel according to brightness ############################
    brightness = np.zeros(len(pure_pix));
    u_max = U_mat.max(axis=0);
    v_max = V_mat.max(axis=0);
    brightness = u_max * v_max;
    brightness_arg = np.argsort(-brightness); #
    brightness_rank = U_mat.shape[1] - rankdata(brightness,method="ordinal");
    U_mat = U_mat[:,brightness_arg];
    V_mat = V_mat[:,brightness_arg];
    temp = np.sqrt((U_mat**2).sum(axis=0,keepdims=True));
    V_mat = V_mat*temp
    U_mat = U_mat/temp;
    if more:
        normalize_factor = np.std(Yd, axis=1, keepdims=True)*T;
        B_mat = np.median(Yd, axis=1, keepdims=True);
        return U_mat, V_mat, B_mat, normalize_factor, brightness_rank
    else:
        return U_mat, V_mat, brightness_rank


def make_mask(corr_img_all_r, corr, mask_a, num_plane=1,times=10,max_allow_neuron_size=0.2):
    """
    update the spatial support: connected region in corr_img(corr(Y,c)) which is connected with previous spatial support
    """
    s = np.ones([3,3]);
    unit_length = int(mask_a.shape[0]/num_plane);
    dims = corr_img_all_r.shape;
    corr_img_all_r = corr_img_all_r.reshape(dims[0],int(dims[1]/num_plane),num_plane,-1,order="F");
    mask_a = mask_a.reshape(corr_img_all_r.shape,order="F");
    corr_ini = corr;
    for ii in range(mask_a.shape[-1]):
        for kk in range(num_plane):
            jj=0;
            corr = corr_ini;
            if mask_a[:,:,kk,ii].sum()>0:
                while jj<=times:
                    labeled_array, num_features = scipy.ndimage.measurements.label(corr_img_all_r[:,:,kk,ii] > corr,structure=s);
                    u, indices, counts = np.unique(labeled_array*mask_a[:,:,kk,ii], return_inverse=True, return_counts=True);
                    if len(u)==1:
                        labeled_array = np.zeros(labeled_array.shape);
                        if corr == 0 or corr == 1:
                            break;
                        else:
                            corr = np.maximum(0, corr - 0.1);
                            jj = jj+1;
                    else:
                        if num_features>1:
                            c = u[1:][np.argmax(counts[1:])];
                            labeled_array = (labeled_array==c);
                            del(c);

                        if labeled_array.sum()/unit_length < max_allow_neuron_size or corr==1 or corr==0:
                            break;
                        else:
                            corr = np.minimum(1, corr + 0.1);
                            jj = jj+1;
                mask_a[:,:,kk,ii] = labeled_array;
    mask_a = (mask_a*1).reshape(unit_length*num_plane,-1,order="F");
    return mask_a


def delete_comp(a, c, corr_img_all_r, mask_a, num_list, temp, word):
    # delete those zero components
    pos = np.where(temp)[0];
    corr_img_all_r = np.delete(corr_img_all_r, pos, axis=2);
    mask_a = np.delete(mask_a, pos, axis=1);
    a = np.delete(a, pos, axis=1);
    c = np.delete(c, pos, axis=1);
    num_list = np.delete(num_list, pos);
    return a, c, corr_img_all_r, mask_a, num_list


def order_superpixels(permute_col, unique_pix, U_mat, V_mat):
    """
    order superpixels according to brightness
    """
    ####################### pull out all the superpixels ################################
    permute_col = list(permute_col);
    pos = [permute_col.index(x) for x in unique_pix];
    U_mat = U_mat[:,pos];
    V_mat = V_mat[:,pos];
    ####################### order pure superpixel according to brightness ############################
    brightness = np.zeros(len(unique_pix));
    u_max = U_mat.max(axis=0);
    v_max = V_mat.max(axis=0);
    brightness = u_max * v_max;
    brightness_rank = U_mat.shape[1] - rankdata(brightness,method="ordinal");
    return brightness_rank


def merge_components_Y(a,c,corr_img_all_r,U,normalize_factor,num_list,patch_size,merge_corr_thr=0.5,merge_overlap_thr=0.8):
    ############ calculate overlap area ###########
    a = csc_matrix(a);
    a_corr = triu(a.T.dot(a),k=1);
    cor = csc_matrix((corr_img_all_r>merge_corr_thr)*1);
    # avoid zero division
    temp = cor.sum(axis=0);
    cor_corr = triu(cor.T.dot(cor),k=1);
    cri = np.asarray((cor_corr/(temp.T)) > merge_overlap_thr)*np.asarray((cor_corr/temp) > merge_overlap_thr)*((a_corr>0).toarray())
    a = a.toarray();
    connect_comps = np.where(cri > 0);
    if len(connect_comps[0]) > 0:
        flag = 1;
        a_pri = a.copy();
        c_pri = c.copy();
        G = nx.Graph();
        G.add_edges_from(list(zip(connect_comps[0], connect_comps[1])))
        comps=list(nx.connected_components(G))
        merge_idx = np.unique(np.concatenate([connect_comps[0], connect_comps[1]],axis=0));
        a_pri = np.delete(a_pri, merge_idx, axis=1);
        c_pri = np.delete(c_pri, merge_idx, axis=1);
        corr_pri = np.delete(corr_img_all_r, merge_idx, axis=1);
        num_pri = np.delete(num_list,merge_idx);
        for comp in comps:
            comp=list(comp);
            a_zero = np.zeros([a.shape[0],1]);
            a_temp = a[:,comp];
            mask_temp = np.where(a_temp.sum(axis=1,keepdims=True) > 0)[0];
            a_temp = a_temp[mask_temp,:];
            y_temp = np.matmul(a_temp, c[:,comp].T);
            a_temp = a_temp.mean(axis=1,keepdims=True);
            c_temp = c[:,comp].mean(axis=1,keepdims=True);
            model = NMF(n_components=1, init='custom')
            a_temp = model.fit_transform(y_temp, W=a_temp, H = (c_temp.T));
            a_zero[mask_temp] = a_temp;
            c_temp = model.components_.T;
            corr_temp = vcorrcoef_Y(U/normalize_factor, c_temp);
            a_pri = np.hstack((a_pri,a_zero));
            c_pri = np.hstack((c_pri,c_temp));
            corr_pri = np.hstack((corr_pri,corr_temp));
            num_pri = np.hstack((num_pri,num_list[comp[0]]));
        return flag, a_pri, c_pri, corr_pri, num_pri
    else:
        flag = 0;
        return flag


def vcorrcoef_Y(U, c):
    """
    fast way to calculate correlation between U and c
    """
    U[np.isnan(U)] = 0;
    temp = (c - c.mean(axis=0,keepdims=True));
    return np.matmul(U - U.mean(axis=1,keepdims=True), temp/np.std(temp, axis=0, keepdims=True));


def ls_solve_ac_Y(X, U, mask=None, beta_LS=None):
    """
    least square solution.
    Parameters:
    ----------------
    X: 2d np.darray
    Y: 2d np.darray
    mask: 2d np.darray
        support constraint of coefficient beta
    ind: 2d binary np.darray
        indication matrix of whether this data is used (=1) or not (=0).
    Return:
    ----------------
    beta_LS: 2d np.darray
        least square solution
    """
    K = X.shape[1];
    if beta_LS is None:
        beta_LS = np.zeros([K,U.shape[1]]);
    UK = np.matmul(X.T, U);
    VK = np.matmul(X.T, X);
    aa = np.diag(VK);
    beta_LS = beta_LS.T;
    for ii in range(K):
        if mask is None:
            beta_LS[[ii],:] = np.maximum(0, beta_LS[[ii],:] + ((UK[[ii],:] - np.matmul(VK[[ii],:],beta_LS))/aa[ii]));
        else:
            ind = (mask[ii,:]>0);
            beta_LS[[ii],ind] = np.maximum(0, beta_LS[[ii],ind] + ((UK[[ii],ind] - np.matmul(VK[[ii],:],beta_LS[:,ind]))/aa[ii]));
    return beta_LS


def ls_solve_acc_Y(X, U, mask=None, beta_LS=None):
    """
    least square solution.
    Parameters:
    ----------------
    X: 2d np.darray
    U: 2d np.darray
    mask: 2d np.darray
        support constraint of coefficient beta
    ind: 2d binary np.darray
        indication matrix of whether this data is used (=1) or not (=0).
    Return:
    ----------------
    beta_LS: 2d np.darray
        least square solution
    """
    K = X.shape[1];
    if beta_LS is None:
        beta_LS = np.zeros([K,U.shape[1]]);
    UK = np.matmul(X.T, U);
    VK = np.matmul(X.T, X);
    aa = np.diag(VK);
    beta_LS = beta_LS.T;
    for ii in range(K):
        if ii<K-1:
            beta_LS[[ii],:] = np.maximum(0, beta_LS[[ii],:] + ((UK[[ii],:] - np.matmul(VK[[ii],:],beta_LS))/aa[ii]));
        else:
            beta_LS[[ii],:] = beta_LS[[ii],:] + ((UK[[ii],:] - np.matmul(VK[[ii],:],beta_LS))/aa[ii]);
    return beta_LS


def update_AC_l2_Y(U, normalize_factor, a, c, b, patch_size, corr_th_fix,
            maxiter=50, tol=1e-8, update_after=None,merge_corr_thr=0.5,
            merge_overlap_thr=0.7, num_plane=1, plot_en=False, max_allow_neuron_size=0.2):
    if U.ndim>2:
        from scipy.sparse import csr_matrix
        dims = U.shape
        T = dims[-1];
        U = U.reshape(np.prod(dims[:-1]),T,order = "F")
    K = c.shape[1];
    res = np.zeros(maxiter);
    uv_mean = U.mean(axis=1,keepdims=True);
    ## initialize spatial support ##
    mask_a = (a>0)*1;
    corr_img_all = vcorrcoef_Y(U/normalize_factor, c);
    corr_img_all_r = corr_img_all.reshape(patch_size[0],patch_size[1],-1,order="F");
    f = np.ones([c.shape[0],1]);
    num_list = np.arange(K);
    for iters in range(maxiter):
        start = time.time();
        print(f'Executing #{iters} iter')
        a = ls_solve_ac_Y(c, (U-b).T, mask=mask_a.T, beta_LS=a).T;
        temp = (a.sum(axis=0) == 0);
        if sum(temp):
            a, c, corr_img_all_r, mask_a, num_list = delete_comp(a, c, corr_img_all_r, mask_a, num_list, temp, "zero a!");
        b = np.maximum(0, uv_mean-((a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)));
        c = ls_solve_ac_Y(a, U-b, mask=None, beta_LS=c).T;
        temp = (c.sum(axis=0) == 0);
        if sum(temp):
            a, c, corr_img_all_r, mask_a, num_list = delete_comp(a, c, corr_img_all_r, mask_a, num_list, temp, "zero c!");
        b = np.maximum(0, uv_mean-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));
        if update_after and ((iters+1) % update_after == 0):
            corr_img_all = vcorrcoef_Y(U/normalize_factor, c);
            rlt = merge_components_Y(a,c,corr_img_all, U, normalize_factor,num_list,patch_size,merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr);
            flag = isinstance(rlt, int);
            if ~np.array(flag):
                a = rlt[1];
                c = rlt[2];
                corr_img_all = rlt[3];
                num_list = rlt[4];
            mask_a = (a>0)*1;
            corr_img_all_r = corr_img_all.reshape(patch_size[0],patch_size[1],-1,order="F");
            mask_a = make_mask(corr_img_all_r, corr_th_fix, mask_a, num_plane, max_allow_neuron_size=max_allow_neuron_size);

            temp = (mask_a.sum(axis=0) == 0);
            if sum(temp):
                a, c, corr_img_all_r, mask_a, num_list = delete_comp(a, c, corr_img_all_r, mask_a, num_list, temp, "zero mask!");
            a = a*mask_a;
        print("time: " + str(time.time()-start))
    temp = np.sqrt((a**2).sum(axis=0,keepdims=True));
    c = c*temp;
    a = a/temp;
    brightness = np.zeros(a.shape[1]);
    a_max = a.max(axis=0);
    c_max = c.max(axis=0);
    brightness = a_max * c_max;
    brightness_rank = np.argsort(-brightness);
    a = a[:,brightness_rank];
    c = c[:,brightness_rank];
    corr_img_all_r = corr_img_all_r[:,:,brightness_rank];
    num_list = num_list[brightness_rank];
    ff = None;
    fb = None;
    return a, c, b, fb, ff, res, corr_img_all_r, num_list


def update_AC_bg_l2_Y(U, normalize_factor, a, c, b, ff, fb, patch_size, corr_th_fix,
            maxiter=50, tol=1e-8, update_after=None,merge_corr_thr=0.5,
            merge_overlap_thr=0.7, num_plane=1, plot_en=False,
            max_allow_neuron_size=0.2):

    if U.ndim>2:
        from scipy.sparse import csr_matrix
        dims = U.shape
        T = dims[-1];
        U = U.reshape(np.prod(dims[:-1]),T,order = "F")

    K = c.shape[1];
    res = np.zeros(maxiter);
    uv_mean = U.mean(axis=1,keepdims=True);
    num_list = np.arange(K);

    num_bg = ff.shape[1];
    f = np.ones([c.shape[0],1]);
    fg = np.ones([a.shape[0],num_bg]);

    ## initialize spatial support ##
    mask_a = (a>0)*1;
    corr_img_all = vcorrcoef_Y(U/normalize_factor, c);
    corr_img_all_r = corr_img_all.reshape(patch_size[0],patch_size[1],-1,order="F");
    mask_ab = np.hstack((mask_a,fg));
    for iters in range(maxiter):
        start = time.time();
        print(f'Executing #{iters} iter')
        temp = ls_solve_ac_Y(np.hstack((c,ff)), (U-b).T, mask=mask_ab.T, beta_LS=np.hstack((a,fb))).T;
        a = temp[:,:-num_bg];
        fb = temp[:,-num_bg:];

        temp = (a.sum(axis=0) == 0);
        if sum(temp):
            a, c, corr_img_all_r, mask_a, num_list = delete_comp(a, c, corr_img_all_r, mask_a, num_list, temp, "zero a!");
        b = np.maximum(0, uv_mean-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));

        temp = ls_solve_acc_Y(np.hstack((a,fb)), U-b, mask=None, beta_LS=np.hstack((c,ff))).T;
        c = temp[:,:-num_bg];
        ff = temp[:,-num_bg:];
        ff = ff - ff.mean(axis=0,keepdims=True);

        temp = (c.sum(axis=0) == 0);
        if sum(temp):
            a, c, corr_img_all_r, mask_a, num_list = delete_comp(a, c, corr_img_all_r, mask_a, num_list, temp, "zero c!");

        b = np.maximum(0, uv_mean-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));

        if update_after and ((iters+1) % update_after == 0):
            corr_img_all = vcorrcoef_Y(U/normalize_factor, c);
            rlt = merge_components_Y(a,c,corr_img_all, U, normalize_factor,num_list,patch_size,merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr);
            flag = isinstance(rlt, int);
            if ~np.array(flag):
                a = rlt[1];
                c = rlt[2];
                corr_img_all = rlt[3];
                num_list = rlt[4];
            mask_a = (a>0)*1;
            corr_img_all_r = corr_img_all.reshape(patch_size[0],patch_size[1],-1,order="F");
            mask_a = make_mask(corr_img_all_r, corr_th_fix, mask_a, num_plane, max_allow_neuron_size=max_allow_neuron_size);

            temp = (mask_a.sum(axis=0) == 0);
            if sum(temp):
                a, c, corr_img_all_r, mask_a, num_list = delete_comp(a, c, corr_img_all_r, mask_a, num_list, temp, "zero mask!");
            a = a*mask_a;
            mask_ab = np.hstack((mask_a,fg));
        print("time: " + str(time.time()-start))
    temp = np.sqrt((a**2).sum(axis=0,keepdims=True));
    c = c*temp;
    a = a/temp;
    brightness = np.zeros(a.shape[1]);
    a_max = a.max(axis=0);
    c_max = c.max(axis=0);
    brightness = a_max * c_max;
    brightness_rank = np.argsort(-brightness);
    a = a[:,brightness_rank];
    c = c[:,brightness_rank];
    corr_img_all_r = corr_img_all_r[:,:,brightness_rank];
    num_list = num_list[brightness_rank];
    return a, c, b, fb, ff, res, corr_img_all_r, num_list


def reconstruct(Yd, spatial_components, temporal_components, background_components, fb=None, ff=None):
    dims = Yd.shape;
    ss = csr_matrix(spatial_components)
    st = csr_matrix(temporal_components)
    print('Compute reconstruction')
    try:
        recon_ = ss.dot(st.T).toarray()+background_components
    except:
        np.savez('error', spatial_components=spatial_components, temporal_components=temporal_components, background_components=background_components)
    if fb is not None:
        recon_ = recon_ + csr_matrix(fb).dot(csr_matrix(ff).T).toarray()
    print('Return reconstruction')
    return Yd - np.asarray(recon_).reshape(dims, order='F')


def demix_whole_data(Yd, cut_off_point=[0.95,0.9], length_cut=[15,10], th=[2,1], pass_num=1, residual_cut = [0.6,0.6],
                    corr_th_fix=0.31, max_allow_neuron_size=0.3, merge_corr_thr=0.6, merge_overlap_thr=0.6, num_plane=1, patch_size=[100,100],
                    plot_en=False, TF=False, fudge_factor=1, text=True, bg=False, max_iter=35, max_iter_fin=50,
                    update_after=4):
    """
    This function is the demixing pipeline for whole data.
    For parameters and output, please refer to demix function (demixing pipeline for low rank data).
    """
    Yd_min = Yd.min();
    if Yd_min < 0:
        Yd_min_pw = Yd.min(axis=-1, keepdims=True);
        Yd -= Yd_min_pw;

    dims = Yd.shape[:2]
    T = Yd.shape[2]
    superpixel_rlt = []
    ## cut image into small parts to find pure superpixels ##
    patch_height = patch_size[0];
    patch_width = patch_size[1];
    height_num = int(np.ceil(dims[0]/patch_height));
    width_num = int(np.ceil(dims[1]/(patch_width*num_plane)));
    num_patch = height_num*width_num;
    patch_ref_mat = np.array(range(num_patch)).reshape(height_num, width_num, order="F");

    for ii in range(pass_num):
        print(f"Execute #{ii} pass........");
        if ii > 0:
            if bg:
                Yd_res = reconstruct(Yd, a, c, b, fb, ff);
            else:
                Yd_res = reconstruct(Yd, a, c, b);
            Yt = threshold_data(Yd_res, th=th[ii]);
        else:
            if th[ii] >= 0:
                Yt = threshold_data(Yd, th=th[ii]);
            else:
                Yt = Yd.copy();
        print("Get threshould data.....")
        start = time.time();
        if num_plane > 1:
            connect_mat_1, idx, comps, permute_col = find_superpixel_3d(Yt,num_plane,cut_off_point[ii],length_cut[ii]);
        else:
            connect_mat_1, idx, comps, permute_col = find_superpixel(Yt,cut_off_point[ii],length_cut[ii]);
        print("time: " + str(time.time()-start));
        if idx==0:
            continue
        start = time.time();
        print("Initialize A and C components....")
        if ii > 0:
            c_ini, a_ini, _, _ = spatial_temporal_ini(Yt, comps, idx, length_cut[ii], bg=False);
        else:
            c_ini, a_ini, ff, fb = spatial_temporal_ini(Yt, comps, idx, length_cut[ii], bg=bg);
        print("time: " + str(time.time()-start));
        unique_pix = np.asarray(np.sort(np.unique(connect_mat_1)),dtype="int");
        unique_pix = unique_pix[np.nonzero(unique_pix)];
        brightness_rank_sup = order_superpixels(permute_col, unique_pix, a_ini, c_ini);
        pure_pix = [];
        start = time.time();
        print("Find pure superpixels....")
        for kk in range(num_patch):
            pos = np.where(patch_ref_mat==kk);
            up=pos[0][0]*patch_height;
            down=min(up+patch_height, dims[0]);
            left=pos[1][0]*patch_width;
            right=min(left+patch_width, dims[1]);
            unique_pix_temp, M = search_superpixel_in_range((connect_mat_1.reshape(dims[0],int(dims[1]/num_plane),num_plane,order="F"))[up:down,left:right], permute_col, c_ini);
            pure_pix_temp = fast_sep_nmf(M, M.shape[1], residual_cut[ii]);
            if len(pure_pix_temp)>0:
                pure_pix = np.hstack((pure_pix, unique_pix_temp[pure_pix_temp]));
        pure_pix = np.unique(pure_pix);
        print("time: " + str(time.time()-start));
        start = time.time();
        print("Prepare iterations....")
        if ii > 0:
            a_ini, c_ini, brightness_rank = prepare_iteration(Yd_res, connect_mat_1, permute_col, pure_pix, a_ini, c_ini);
            a = np.hstack((a, a_ini));
            c = np.hstack((c, c_ini));
        else:
            a, c, b, normalize_factor, brightness_rank = prepare_iteration(Yd, connect_mat_1, permute_col, pure_pix, a_ini, c_ini, more=True);
        print("time: " + str(time.time()-start));
        if a.size==0:
            continue
        if ii == pass_num - 1:
            maxiter = max_iter_fin;
        else:
            maxiter=max_iter;
        if bg:
            a, c, b, fb, ff, res, corr_img_all_r, num_list = update_AC_bg_l2_Y(Yd.reshape(np.prod(dims),-1,order="F"), normalize_factor, a, c, b, ff, fb, dims,
                                        corr_th_fix, maxiter=maxiter, tol=1e-8, update_after=update_after,
                                        merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr, num_plane=num_plane, max_allow_neuron_size=max_allow_neuron_size);
        else:
            a, c, b, fb, ff, res, corr_img_all_r, num_list = update_AC_l2_Y(Yd.reshape(np.prod(dims),-1,order="F"), normalize_factor, a, c, b, dims,
                                        corr_th_fix, maxiter=maxiter, tol=1e-8, update_after=update_after,
                                        merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr, num_plane=num_plane, max_allow_neuron_size=max_allow_neuron_size);
        superpixel_rlt.append({'connect_mat_1':connect_mat_1, 'pure_pix':pure_pix, 'unique_pix':unique_pix, 'brightness_rank':brightness_rank, 'brightness_rank_sup':brightness_rank_sup});
        if pass_num > 1 and ii == 0:
            rlt = {'a':a, 'c':c, 'b':b, "fb":fb, "ff":ff};

    if (idx==0) & (ii==0):
        fin_rlt = {'a':np.zeros((np.prod(dims[:-1]), 1))};
    else:
        fin_rlt = {'a':a, 'c':c, 'b':b, "fb":fb, "ff":ff};
    if pass_num > 1:
        return {'rlt':rlt, 'fin_rlt':fin_rlt, "superpixel_rlt":superpixel_rlt}
    else:
        return {'fin_rlt':fin_rlt, "superpixel_rlt":superpixel_rlt}
