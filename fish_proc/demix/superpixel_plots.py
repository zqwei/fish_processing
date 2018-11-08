import numpy as np
import matplotlib.pyplot as plt


def match_comp(rlt, rlt_lasso_Ydc, rlt_lasso_Yrawc, rlt_a, rlt_lasso_Yda, rlt_lasso_Yrawa,th):
    K = rlt.shape[1];
    order_Yd = np.zeros([K])
    order_Yraw = np.zeros([K])
    for ii in range(K):
        temp = vcorrcoef2(rlt_lasso_Ydc.T, rlt[:,ii]);
        temp2 = vcorrcoef2(rlt_lasso_Yrawc.T, rlt[:,ii]);
        pos = np.argsort(-temp)[:sum(temp > th)];
        pos2 = np.argsort(-temp2)[:sum(temp2 > th)];

        if len(pos)>0:
            spa_temp = np.where(np.matmul(rlt_a[:,[ii]].T, rlt_lasso_Yda[:,pos])>0)[1];
            if len(spa_temp)>0:
                order_Yd[ii] = int(pos[spa_temp[0]]);
            else:
                order_Yd[ii] = np.nan;
        else:
            order_Yd[ii] = np.nan;

        if len(pos2)>0:
            spa_temp2 = np.where(np.matmul(rlt_a[:,[ii]].T, rlt_lasso_Yrawa[:,pos2])>0)[1];
            if len(spa_temp2)>0:
                order_Yraw[ii] = int(pos2[spa_temp2[0]]);
            else:
                order_Yraw[ii] = np.nan;
        else:
            order_Yraw[ii] = np.nan;
    order_Yd = np.asarray(order_Yd,dtype=int);
    order_Yraw = np.asarray(order_Yraw,dtype=int);
    return order_Yd, order_Yraw


def match_comp_gt(rlt_gt, rlt, rlt_lasso_Ydc, rlt_lasso_Yrawc,rlt_gta, rlt_a, rlt_lasso_Yda, rlt_lasso_Yrawa,th):
    K = rlt_gt.shape[1];
    order_Ys = np.zeros([K]);
    order_Yd = np.zeros([K])
    order_Yraw = np.zeros([K])
    for ii in range(K):
        temp0 = vcorrcoef2(rlt.T, rlt_gt[:,ii]);
        temp = vcorrcoef2(rlt_lasso_Ydc.T, rlt_gt[:,ii]);
        temp2 = vcorrcoef2(rlt_lasso_Yrawc.T, rlt_gt[:,ii]);
        pos0 = np.argsort(-temp0)[:sum(temp0 > th)];
        pos = np.argsort(-temp)[:sum(temp > th)];
        pos2 = np.argsort(-temp2)[:sum(temp2 > th)];

        if len(pos0)>0:
            spa_temp0 = np.where(np.matmul(rlt_gta[:,[ii]].T, rlt_a[:,pos0])>0)[1];
            if len(spa_temp0)>0:
                #print(int(pos0[spa_temp0]));
                order_Ys[ii] = int(pos0[spa_temp0[0]]);
                if (order_Ys[:ii]==int(pos0[spa_temp0[0]])).sum()>0:
                    order_Ys[ii] = np.nan;
            else:
                order_Ys[ii] = np.nan;
            #if ii == K-1:
            #	order_Ys[ii] = 13;
        else:
            order_Ys[ii] = np.nan;
        if len(pos)>0:
            spa_temp = np.where(np.matmul(rlt_gta[:,[ii]].T, rlt_lasso_Yda[:,pos])>0)[1];
            if len(spa_temp)>0:
                order_Yd[ii] = int(pos[spa_temp[0]]);
                if (order_Yd[:ii]==int(pos[spa_temp[0]])).sum()>0:
                    order_Yd[ii] = np.nan;
            else:
                order_Yd[ii] = np.nan;
        else:
            order_Yd[ii] = np.nan;

        if len(pos2)>0:
            spa_temp2 = np.where(np.matmul(rlt_gta[:,[ii]].T, rlt_lasso_Yrawa[:,pos2])>0)[1];
            if len(spa_temp2)>0:
                order_Yraw[ii] = int(pos2[spa_temp2[0]]);
                if (order_Yraw[:ii]==int(pos2[spa_temp2[0]])).sum()>0:
                    order_Yraw[ii] = np.nan;
            else:
                order_Yraw[ii] = np.nan;
        else:
            order_Yraw[ii] = np.nan;
    order_Ys = np.asarray(order_Ys,dtype=int);
    order_Yd = np.asarray(order_Yd,dtype=int);
    order_Yraw = np.asarray(order_Yraw,dtype=int);
    return order_Ys, order_Yd, order_Yraw


def match_comp_projection(rlt_xyc, rlt_yzc, rlt_xya, rlt_yza, dims1, dims2, th):
    K = rlt_xyc.shape[1];
    order = np.zeros([K]);
    rlt_xya = rlt_xya.reshape(dims1[0],dims1[1],-1,order="F");
    rlt_yza = rlt_yza.reshape(dims2[0],dims2[1],-1,order="F");

    for ii in range(K):
        temp0 = vcorrcoef2(rlt_yzc.T, rlt_xyc[:,ii]);
        pos0 = np.argsort(-temp0)[:sum(temp0 > th)];

        if len(pos0)>0:
            spa_temp0 = np.where(np.matmul(rlt_xya[:,:,[ii]].sum(axis=0).T, rlt_yza[:,:,pos0].sum(axis=0))>0)[1];
            #print(spa_temp0);
            if len(spa_temp0)>0:
                #print(int(pos0[spa_temp0]));
                order[ii] = int(pos0[spa_temp0[0]]);
            else:
                order[ii] = np.nan;
        else:
            order[ii] = np.nan;
    order = np.asarray(order,dtype=int);
    return order


def superpixel_single_plot(connect_mat_1,unique_pix,brightness_rank_sup,text):
    scale = np.maximum(1, (connect_mat_1.shape[1]/connect_mat_1.shape[0]));
    fig = plt.figure(figsize=(4*scale,4));
    ax = plt.subplot(1,1,1);
    ax.imshow(connect_mat_1,cmap="nipy_spectral_r");

    if text:
        for ii in range(len(unique_pix)):
            pos = np.where(connect_mat_1[:,:] == unique_pix[ii]);
            pos0 = pos[0];
            pos1 = pos[1];
            ax.text((pos1)[np.array(len(pos1)/3,dtype=int)], (pos0)[np.array(len(pos0)/3,dtype=int)], f"{brightness_rank_sup[ii]+1}",
                verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)#, fontweight="bold")
    ax.set(title="Superpixels")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")
    return fig


def pure_superpixel_single_plot(connect_mat_1,pure_pix,brightness_rank,text,pure=True):
    scale = np.maximum(1, (connect_mat_1.shape[1]/connect_mat_1.shape[0]));
    fig = plt.figure(figsize=(4*scale,4));
    ax1 = plt.subplot(1,1,1);
    dims = connect_mat_1.shape;
    connect_mat_1_pure = connect_mat_1.copy();
    connect_mat_1_pure = connect_mat_1_pure.reshape(np.prod(dims),order="F");
    connect_mat_1_pure[~np.in1d(connect_mat_1_pure,pure_pix)]=0;
    connect_mat_1_pure = connect_mat_1_pure.reshape(dims,order="F");

    ax1.imshow(connect_mat_1_pure,cmap="nipy_spectral_r");

    if text:
        for ii in range(len(pure_pix)):
            pos = np.where(connect_mat_1_pure[:,:] == pure_pix[ii]);
            pos0 = pos[0];
            pos1 = pos[1];
            ax1.text((pos1)[np.array(len(pos1)/3,dtype=int)], (pos0)[np.array(len(pos0)/3,dtype=int)], f"{brightness_rank[ii]+1}",
                verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)#, fontweight="bold")
    if pure:
        ax1.set(title="Pure superpixels");
    else:
        ax1.set(title="Superpixels");
    ax1.title.set_fontsize(15)
    ax1.title.set_fontweight("bold");
    plt.tight_layout();
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    return fig


def pure_superpixel_corr_compare_plot(connect_mat_1, unique_pix, pure_pix, brightness_rank_sup, brightness_rank, Cnt, text=False):
    scale = np.maximum(1, (connect_mat_1.shape[1]/connect_mat_1.shape[0]));
    fig = plt.figure(figsize=(4*scale,12));
    ax = plt.subplot(3,1,1);
    ax.imshow(connect_mat_1,cmap="nipy_spectral_r");

    if text:
        for ii in range(len(unique_pix)):
            pos = np.where(connect_mat_1[:,:] == unique_pix[ii]);
            pos0 = pos[0];
            pos1 = pos[1];
            ax.text((pos1)[np.array(len(pos1)/3,dtype=int)], (pos0)[np.array(len(pos0)/3,dtype=int)], f"{brightness_rank_sup[ii]+1}",
                verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)#, fontweight="bold")
    ax.set(title="Superpixels")
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")

    ax1 = plt.subplot(3,1,2);
    dims = connect_mat_1.shape;
    connect_mat_1_pure = connect_mat_1.copy();
    connect_mat_1_pure = connect_mat_1_pure.reshape(np.prod(dims),order="F");
    connect_mat_1_pure[~np.in1d(connect_mat_1_pure,pure_pix)]=0;
    connect_mat_1_pure = connect_mat_1_pure.reshape(dims,order="F");

    ax1.imshow(connect_mat_1_pure,cmap="nipy_spectral_r");

    if text:
        for ii in range(len(pure_pix)):
            pos = np.where(connect_mat_1_pure[:,:] == pure_pix[ii]);
            pos0 = pos[0];
            pos1 = pos[1];
            ax1.text((pos1)[np.array(len(pos1)/3,dtype=int)], (pos0)[np.array(len(pos0)/3,dtype=int)], f"{brightness_rank[ii]+1}",
                verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)#, fontweight="bold")
    ax1.set(title="Pure superpixels")
    ax1.title.set_fontsize(15)
    ax1.title.set_fontweight("bold");

    ax2 = plt.subplot(3,1,3);
    show_img(ax2, Cnt);
    ax2.set(title="Local mean correlation")
    ax2.title.set_fontsize(15)
    ax2.title.set_fontweight("bold")
    plt.tight_layout()
    plt.show();
    return fig


def show_img(ax, img,vmin=None,vmax=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    im = ax.imshow(img,cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    if np.abs(img.min())< 1:
        format_tile ='%.2f'
    else:
        format_tile ='%5d'
    plt.colorbar(im, cax=cax,orientation='vertical',spacing='uniform')


def temporal_comp_plot(c, num_list=None, ini = False):
    num = c.shape[1];
    fig = plt.figure(figsize=(20,1.5*num))
    if num_list is None:
        num_list = np.arange(num);
    for ii in range(num):
        plt.subplot(num,1, ii+1);
        plt.plot(c[:,ii]);
        if ii == 0:
            if ini:
                plt.title("Temporal components initialization for pure superpixels",fontweight="bold",fontsize=15);
            else:
                plt.title("Temporal components",fontweight="bold",fontsize=15);
        plt.ylabel(f"{num_list[ii]+1}",fontweight="bold",fontsize=15)
        if (ii > 0 and ii < num-1):
            plt.tick_params(axis='x',which='both',labelbottom='off')
        else:
            plt.xlabel("frames");
    plt.tight_layout()
    plt.show()
    return fig


def spatial_comp_plot(a, corr_img_all_r, num_list=None, ini=False):
    num = a.shape[1];
    patch_size = corr_img_all_r.shape[:2];
    scale = np.maximum(1, (corr_img_all_r.shape[1]/corr_img_all_r.shape[0]));
    fig = plt.figure(figsize=(8*scale,4*num));
    if num_list is None:
        num_list = np.arange(num);
    for ii in range(num):
        plt.subplot(num,2,2*ii+1);
        plt.imshow(a[:,ii].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.ylabel(str(num_list[ii]+1),fontsize=15,fontweight="bold");
        if ii==0:
            if ini:
                plt.title("Spatial components ini",fontweight="bold",fontsize=15);
            else:
                plt.title("Spatial components",fontweight="bold",fontsize=15);
        ax1 = plt.subplot(num,2,2*(ii+1));
        show_img(ax1, corr_img_all_r[:,:,ii]);
        if ii==0:
            ax1.set(title="corr image")
            ax1.title.set_fontsize(15)
            ax1.title.set_fontweight("bold")
    plt.tight_layout()
    plt.show()
    return fig


def spatial_sum_plot(a, a_fin, patch_size, num_list_fin=None, text=False):
    scale = np.maximum(1, (patch_size[1]/patch_size[0]));
    fig = plt.figure(figsize=(16*scale,8));
    ax = plt.subplot(1,2,1);
    ax.imshow(a_fin.sum(axis=1).reshape(patch_size,order="F"),cmap="jet");

    if num_list_fin is None:
        num_list_fin = np.arange(a_fin.shape[1]);
    if text:
        for ii in range(a_fin.shape[1]):
            temp = a_fin[:,ii].reshape(patch_size,order="F");
            pos0 = np.where(temp == temp.max())[0][0];
            pos1 = np.where(temp == temp.max())[1][0];
            ax.text(pos1, pos0, f"{num_list_fin[ii]+1}", verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")

    ax.set(title="more passes spatial components")
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")

    ax1 = plt.subplot(1,2,2);
    ax1.imshow(a.sum(axis=1).reshape(patch_size,order="F"),cmap="jet");

    if text:
        for ii in range(a.shape[1]):
            temp = a[:,ii].reshape(patch_size,order="F");
            pos0 = np.where(temp == temp.max())[0][0];
            pos1 = np.where(temp == temp.max())[1][0];
            ax1.text(pos1, pos0, f"{ii+1}", verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")

    ax1.set(title="1 pass spatial components")
    ax1.title.set_fontsize(15)
    ax1.title.set_fontweight("bold")
    plt.tight_layout();
    plt.show()
    return fig


def spatial_sum_plot_single(a_fin, patch_size, num_list_fin=None, text=False):
    scale = np.maximum(1, (patch_size[1]/patch_size[0]));
    fig = plt.figure(figsize=(4*scale,4));
    ax = plt.subplot(1,1,1);
    ax.imshow(a_fin.sum(axis=1).reshape(patch_size,order="F"),cmap="nipy_spectral_r");

    if num_list_fin is None:
        num_list_fin = np.arange(a_fin.shape[1]);
    if text:
        for ii in range(a_fin.shape[1]):
            temp = a_fin[:,ii].reshape(patch_size,order="F");
            pos0 = np.where(temp == temp.max())[0][0];
            pos1 = np.where(temp == temp.max())[1][0];
            ax.text(pos1, pos0, f"{num_list_fin[ii]+1}", verticalalignment='bottom', horizontalalignment='right',color='black', fontsize=15)

    ax.set(title="Cumulative spatial components")
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")

    plt.tight_layout();
    plt.show()
    return fig


def spatial_match_projection_plot(order, number, rlt_xya, rlt_yza, dims1, dims2):
    number = (order>=0).sum();
    scale = (dims1[1]+dims2[1])/max(dims1[0],dims2[0]);
    fig = plt.figure(figsize=(scale*2, 2*number));
    temp0 = np.where(order>=0)[0];
    temp1 = order[temp0];
    for ii in range(number):
        plt.subplot(number,2,2*ii+1);
        plt.imshow(rlt_xya[:,temp0[ii]].reshape(dims1[:2],order="F"),cmap="jet",aspect="auto");
        if ii == 0:
            plt.title("xy",fontsize=15,fontweight="bold");
            plt.ylabel("x",fontsize=15,fontweight="bold");
            plt.xlabel("y",fontsize=15,fontweight="bold");

        plt.subplot(number,2,2*ii+2);
        plt.imshow(rlt_yza[:,temp1[ii]].reshape(dims2[:2],order="F"),cmap="jet",aspect="auto");
        if ii == 0:
            plt.title("zy",fontsize=15,fontweight="bold");
            plt.ylabel("z",fontsize=15,fontweight="bold");
            plt.xlabel("y",fontsize=15,fontweight="bold");
    plt.tight_layout()
    return fig


def spatial_compare_single_plot(a, patch_size):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    scale = (patch_size[1]/patch_size[0]);
    fig = plt.figure(figsize=(4*scale,4));
    ax1 = plt.subplot(1,1,1);
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    img1 = ax1.imshow(a.reshape(patch_size,order="F"),cmap='nipy_spectral_r');
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(img1, cax=cax,orientation='vertical',spacing='uniform')
    plt.tight_layout();
    plt.show();
    return fig


def spatial_compare_nmf_plot(a, a_lasso_den, a_lasso_raw, order_Yd, order_Yraw, patch_size):
    num = a.shape[1];
    scale = (patch_size[1]/patch_size[0]);
    fig = plt.figure(figsize=(12*scale,4*num));

    for ii in range(num):
        ax0=plt.subplot(num,3,3*ii+1);
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        img0=plt.imshow(a[:,ii].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        if ii==0:
            plt.title("Our method",fontweight="bold",fontsize=15);

        ax1=plt.subplot(num,3,3*ii+2);
        if ii==0:
            plt.title("Sparse nmf on denoised data",fontweight="bold",fontsize=15);
        if order_Yd[ii]>=0:
            img1=plt.imshow(a_lasso_den[:,order_Yd[ii]].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

        ax2=plt.subplot(num,3,3*ii+3);
        if ii==0:
            plt.title("Sparse nmf on raw data",fontweight="bold",fontsize=15);
        if order_Yraw[ii]>=0:
            img2=plt.imshow(a_lasso_raw[:,order_Yraw[ii]].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    plt.tight_layout()
    plt.show()
    return fig


def spatial_compare_nmf_gt_plot(a_gt, a, a_lasso_den, a_lasso_raw, order_Ys, order_Yd, order_Yraw, patch_size):
    num = a_gt.shape[1];
    scale = np.maximum(1, (patch_size[1]/patch_size[0]));
    fig = plt.figure(figsize=(16*scale,4*num));

    for ii in range(num):
        ax00=plt.subplot(num,4,4*ii+1);
        img00=plt.imshow(a_gt[:,ii].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        if ii==0:
            plt.title("Ground truth",fontweight="bold",fontsize=15);

        ax0=plt.subplot(num,4,4*ii+2);
        if order_Ys[ii]>=0:
            img0=plt.imshow(a[:,order_Ys[ii]].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        if ii==0:
            plt.title("Our method",fontweight="bold",fontsize=15);

        ax1=plt.subplot(num,4,4*ii+3);
        if ii==0:
            plt.title("Sparse nmf on denoised data",fontweight="bold",fontsize=15);
        if order_Yd[ii]>=0:
            img1=plt.imshow(a_lasso_den[:,order_Yd[ii]].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

        ax2=plt.subplot(num,4,4*ii+4);
        if ii==0:
            plt.title("Sparse nmf on raw data",fontweight="bold",fontsize=15);
        if order_Yraw[ii]>=0:
            img2=plt.imshow(a_lasso_raw[:,order_Yraw[ii]].reshape(patch_size,order="F"),cmap='nipy_spectral_r');
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    plt.tight_layout()
    plt.show()
    return fig


def temporal_compare_nmf_plot(c, c_lasso_den, c_lasso_raw, order_Yd, order_Yraw):
    num = c.shape[1];
    fig = plt.figure(figsize=(20,1.5*num))
    for ii in range(num):
        plt.subplot(num,1, ii+1);
        plt.plot(c[:,ii],label="our method");
        if order_Yd[ii]>=0:
            plt.plot(c_lasso_den[:,order_Yd[ii]],label="sparse nmf on denoised data");
        if order_Yraw[ii]>=0:
            plt.plot(c_lasso_raw[:,order_Yraw[ii]],label="sparse nmf on raw data");
        plt.legend();
        if ii == 0:
            plt.title("Temporal components",fontweight="bold",fontsize=15);
        plt.ylabel(f"{ii+1}",fontweight="bold",fontsize=15)
        if (ii > 0 and ii < num-1):
            plt.tick_params(axis='x',which='both',labelbottom='off')
        else:
            plt.xlabel("frames");
    plt.tight_layout()
    plt.show()
    return fig


def temporal_compare_plot(c, c_tf, ini = False):
    num = c.shape[1];
    fig = plt.figure(figsize=(20,1.5*num))
    for ii in range(num):
        plt.subplot(num,1, ii+1);
        plt.plot(c[:,ii],label="c");
        plt.plot(c_tf[:,ii],label="c_tf");
        plt.legend();
        if ii == 0:
            if ini:
                plt.title("Temporal components initialization for pure superpixels",fontweight="bold",fontsize=15);
            else:
                plt.title("Temporal components",fontweight="bold",fontsize=15);
        plt.ylabel(f"{ii+1}",fontweight="bold",fontsize=15)
        if (ii > 0 and ii < num-1):
            plt.tick_params(axis='x',which='both',labelbottom='off')
        else:
            plt.xlabel("frames");
    plt.tight_layout()
    plt.show()
    return fig