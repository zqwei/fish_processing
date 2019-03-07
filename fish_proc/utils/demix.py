'''
A set of posthoc processing from demix algorithm
------------------------
Ziqiang Wei @ 2018
weiz@janelia.hhmi.org
'''

from ..demix.superpixel_analysis import *
import numpy as np

def pos_sig_correction(mov, axis_):
    return mov - (mov).min(axis=axis_, keepdims=True)

def recompute_C_matrix_sparse(sig, A):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    d1, d2, T = sig.shape
    sig = csr_matrix(np.reshape(sig, (d1*d2,T), order='F'))
    A = csr_matrix(A)
    return np.asarray(spsolve(A, sig).todense())

def recompute_C_matrix(sig, A, issparse=False):
    if not issparse:
        d1, d2, T = sig.shape
        return np.linalg.inv(np.array(A.T.dot(A))).dot(A.T.dot(np.reshape(sig, (d1*d2,T), order='F')))
    else:
        return recompute_C_matrix_sparse(sig, A)

def recompute_nmf(rlt_, mov, comp_thres=0):
    b = rlt_['fin_rlt']['b']
    fb = rlt_['fin_rlt']['fb']
    ff = rlt_['fin_rlt']['ff']
    dims = mov.shape
    if fb is not None:
        b_ = np.matmul(fb, ff.T)+b
    else:
        b_ = b
    mov_pos = pos_sig_correction(mov, -1)
    mov_no_background = mov_pos - b_.reshape((dims[0], dims[1], len(b_)//dims[0]//dims[1]), order='F')
    A = rlt_['fin_rlt']['a']
    A = A[:, (A>0).sum(axis=0)>comp_thres]
    C_ = recompute_C_matrix(mov_no_background, A)
    mov_res = reconstruct(mov_pos, A, C_.T, b_, fb=fb, ff=ff)
    mov_res_ = mov_res.mean(axis=-1, keepdims=True)
    b_ = b_.reshape((dims[0], dims[1], len(b_)//dims[0]//dims[1]), order='F')
    return C_, b_+mov_res_, mov_res-mov_res_

def compute_res(mov_pos, rlt_):
    return reconstruct(mov_pos, rlt_['fin_rlt']['a'], rlt_['fin_rlt']['c'],
                       rlt_['fin_rlt']['b'], fb=rlt_['fin_rlt']['fb'], ff=rlt_['fin_rlt']['ff'])

def demix_whole_data_snr(Yd, cut_off_point=[0.95,0.9], length_cut=[15,10],
                        th=[2,1], pass_num=1, residual_cut = [0.6,0.6],
                        corr_th_fix=0.31, max_allow_neuron_size=0.3,
                        merge_corr_thr=0.6, merge_overlap_thr=0.6, num_plane=1,
                        patch_size=[100,100], std_thres=0.5, plot_en=False, TF=False,
                        fudge_factor=1, text=True, bg=False, max_iter=35,
                        max_iter_fin=50, update_after=4):
    """
    This function is the demixing pipeline for whole data.
    For parameters and output, please refer to demix function (demixing pipeline for low rank data).
    """
    ## if data has negative values then do pixel-wise minimum subtraction ##
    Yd_min = Yd.min();
    # threshold data using its variability
    Y_amp  = Yd.std(axis=-1)
    if Yd_min < 0:
        Yd_min_pw = Yd.min(axis=2, keepdims=True);
        Yd -= Yd_min_pw;

    dims = Yd.shape[:2];
    T = Yd.shape[2];
    superpixel_rlt = [];
    ## cut image into small parts to find pure superpixels ##

    patch_height = patch_size[0];
    patch_width = patch_size[1];
    height_num = int(np.ceil(dims[0]/patch_height));  ########### if need less data to find pure superpixel, change dims[0] here #################
    width_num = int(np.ceil(dims[1]/(patch_width*num_plane)));
    num_patch = height_num*width_num;
    patch_ref_mat = np.array(range(num_patch)).reshape(height_num, width_num, order="F");

    ii = 0;
    while ii < pass_num:
        print("start " + str(ii+1) + " pass!");
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

        Yt_ = Yt.copy()
        Yt_[Y_amp<std_thres] += np.random.normal(size=Yt.shape)[Y_amp<std_thres]
        start = time.time();
        if num_plane > 1:
            print("3d data!");
            connect_mat_1, idx, comps, permute_col = find_superpixel_3d(Yt_,num_plane,cut_off_point[ii],length_cut[ii],eight_neighbours=True);
        else:
            print("find superpixels!")
            connect_mat_1, idx, comps, permute_col = find_superpixel(Yt_,cut_off_point[ii],length_cut[ii],eight_neighbours=True);
        print("time: " + str(time.time()-start));

        start = time.time();
        print("rank 1 svd!")
        if ii > 0:
            c_ini, a_ini, _, _ = spatial_temporal_ini(Yt, comps, idx, length_cut[ii], bg=False);
        else:
            c_ini, a_ini, ff, fb = spatial_temporal_ini(Yt, comps, idx, length_cut[ii], bg=bg);
            #return ff
        print("time: " + str(time.time()-start));
        unique_pix = np.asarray(np.sort(np.unique(connect_mat_1)),dtype="int");
        unique_pix = unique_pix[np.nonzero(unique_pix)];
        #unique_pix = np.asarray(np.sort(np.unique(connect_mat_1))[1:]); #search_superpixel_in_range(connect_mat_1, permute_col, V_mat);
        brightness_rank_sup = order_superpixels(permute_col, unique_pix, a_ini, c_ini);

        #unique_pix = np.asarray(unique_pix);
        pure_pix = [];

        start = time.time();
        print("find pure superpixels!")
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
        print("prepare iteration!")
        if ii > 0:
            a_ini, c_ini, brightness_rank = prepare_iteration(Yd_res, connect_mat_1, permute_col, pure_pix, a_ini, c_ini);
            a = np.hstack((a, a_ini));
            c = np.hstack((c, c_ini));
        else:
            a, c, b, normalize_factor, brightness_rank = prepare_iteration(Yd, connect_mat_1, permute_col, pure_pix, a_ini, c_ini, more=True);

        print("time: " + str(time.time()-start));

        if plot_en:
            Cnt = local_correlations_fft(Yt);
            pure_superpixel_corr_compare_plot(connect_mat_1, unique_pix, pure_pix, brightness_rank_sup, brightness_rank, Cnt, text);
        print("start " + str(ii+1) + " pass iteration!")
        if ii == pass_num - 1:
            maxiter = max_iter_fin;
        else:
            maxiter=max_iter;
        start = time.time();
        if bg:
            a, c, b, fb, ff, res, corr_img_all_r, num_list = update_AC_bg_l2_Y(Yd.reshape(np.prod(dims),-1,order="F"), normalize_factor, a, c, b, ff, fb, dims,
                                        corr_th_fix, maxiter=maxiter, tol=1e-8, update_after=update_after,
                                        merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr, num_plane=num_plane, plot_en=plot_en, max_allow_neuron_size=max_allow_neuron_size);

        else:
            a, c, b, fb, ff, res, corr_img_all_r, num_list = update_AC_l2_Y(Yd.reshape(np.prod(dims),-1,order="F"), normalize_factor, a, c, b, dims,
                                        corr_th_fix, maxiter=maxiter, tol=1e-8, update_after=update_after,
                                        merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr, num_plane=num_plane, plot_en=plot_en, max_allow_neuron_size=max_allow_neuron_size);
        print("time: " + str(time.time()-start));
        superpixel_rlt.append({'connect_mat_1':connect_mat_1, 'pure_pix':pure_pix, 'unique_pix':unique_pix, 'brightness_rank':brightness_rank, 'brightness_rank_sup':brightness_rank_sup});
        if pass_num > 1 and ii == 0:
            rlt = {'a':a, 'c':c, 'b':b, "fb":fb, "ff":ff, 'res':res, 'corr_img_all_r':corr_img_all_r, 'num_list':num_list};
            a0 = a.copy();
        ii = ii+1;

    c_tf = [];
    start = time.time();
    if TF:
        sigma = noise_estimator(c.T);
        sigma *= fudge_factor
        for ii in range(c.shape[1]):
            c_tf = np.hstack((c_tf, l1_tf(c[:,ii], sigma[ii])));
        c_tf = c_tf.reshape(T,int(c_tf.shape[0]/T),order="F");
    print("time: " + str(time.time()-start));
    if plot_en:
        if pass_num > 1:
            spatial_sum_plot(a0, a, dims, num_list, text);
        Yd_res = reconstruct(Yd, a, c, b);
        Yd_res = threshold_data(Yd_res, th=0);
        Cnt = local_correlations_fft(Yd_res);
        scale = np.maximum(1, int(Cnt.shape[1]/Cnt.shape[0]));
        plt.figure(figsize=(8*scale,8))
        ax1 = plt.subplot(1,1,1);
        show_img(ax1, Cnt);
        ax1.set(title="Local mean correlation for residual")
        ax1.title.set_fontsize(15)
        ax1.title.set_fontweight("bold")
        plt.show();
    fin_rlt = {'a':a, 'c':c, 'c_tf':c_tf, 'b':b, "fb":fb, "ff":ff, 'res':res, 'corr_img_all_r':corr_img_all_r, 'num_list':num_list};
    if Yd_min < 0:
        Yd += Yd_min_pw;

    if pass_num > 1:
        return {'rlt':rlt, 'fin_rlt':fin_rlt, "superpixel_rlt":superpixel_rlt}
    else:
        return {'fin_rlt':fin_rlt, "superpixel_rlt":superpixel_rlt}
