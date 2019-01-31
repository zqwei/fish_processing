import numpy as np
import multiprocessing
import time
import matplotlib.pyplot as plt
from . import greedyPCA as gpca
from math import ceil
from functools import partial
from itertools import product
from ..utils.memory import get_process_memory, clear_variables

def block_split_size(l,n):
    """
    For an array of length l that should be split into n sections,
    calculate the dimension of each section:
    l%n sub-arrays of size l//n +1 and the rest of size l//n
    Input:
    ------
    l:      int
            length of array
    n:      int
            number of section in which an array of size l
            will be partitioned

    Output:
    ------
    d:      np.array (n,)
            length of each partitioned array.
    """
    d = np.zeros((n,)).astype('int')
    cut = l%n
    d[:cut] = l//n+1
    d[cut:] = l//n
    return d


def split_image_into_blocks(image, nblocks=[10,10]):
    """
    Split an image into blocks.

    Parameters:
    ----------
    image:          np.array (d1 x d2 x T)
                    array to be split into nblocks
                    along first two dimensions
    nblocks:        list (2,)
                    parameters to split image across
                    the first two dimensions, respectively

    Outputs
    -------
    blocks:         list,
                    contains nblocks[0]*nblocks[1] number of tiles
                    each of dimensions (d1' x d2' x T)
                    in fortran 'F' order.
    """
    if all(isinstance(n, int) for n in nblocks):
        number_of_blocks = np.prod(nblocks)
    else:
        number_of_blocks = (len(nblocks[0])+1)*(len(nblocks[1])+1)
    blocks = []
    if number_of_blocks != (image.shape[0] * image.shape[1]):
        block_divided_image = np.array_split(image,nblocks[0],axis=0)
        for row in block_divided_image:
            blocks_ = np.array_split(row,nblocks[1],axis=1)
            for block in blocks_:
                blocks.append(np.array(block))
    else:
        blocks = image.flatten()
    return blocks


def vector_offset(array,offset_factor=2):
    """
    Given the dimenions of a matrix (dims), which was
    split row and column wise according to row_array,col_array,
    Calculate the offset in which to split the
    """
    array_offset = np.ceil(np.divide(np.diff(array),
                                     offset_factor)).astype('int')
    return array_offset


def tile_grids(dims,
               nblocks=[10,10],
               indiv_grids=True):
    if all(isinstance(n, int) for n in nblocks):
        d_row = block_split_size(dims[0],nblocks[0])
        d_col = block_split_size(dims[1],nblocks[1])
    else:
        d_row,d_col=nblocks
    if indiv_grids:
        d_row = np.insert(d_row,0,0)
        d_col = np.insert(d_col,0,0)
        return d_row.cumsum(),d_col.cumsum()
    d_row = np.append(d_row,dims[0])
    d_col = np.append(d_col,dims[1])
    d_row = np.diff(np.insert(d_row,0,0))
    d_col = np.diff(np.insert(d_col,0,0))
    number_of_blocks = (len(d_row))*(len(d_col))
    array = np.zeros((number_of_blocks,2))
    for ii,row in enumerate(product(d_row,d_col)):
        array[ii]=row
    return array.astype('int')


def offset_tiling_dims(dims,nblocks,offset_case=None):
    row_array, col_array = tile_grids(dims,nblocks)
    r_offset = vector_offset(row_array)
    c_offset = vector_offset(col_array)
    rc0, rc1 = (row_array[1:]-r_offset)[[0,-1]]
    cc0, cc1 = (col_array[1:]-c_offset)[[0,-1]]
    if offset_case is None:
        row_array=row_array[1:-1]
        col_array=col_array[1:-1]
    elif offset_case == 'r':
        dims = rc1-rc0,dims[1],dims[2]
        row_array=row_array[1:-2]
        col_array=col_array[1:-1]
    elif offset_case == 'c':
        dims = dims[0],cc1-cc0,dims[2]
        row_array=row_array[1:-1]
        col_array=col_array[1:-2]
    elif offset_case == 'rc':
        dims = rc1-rc0,cc1-cc0,dims[2]
        row_array=row_array[1:-2]
        col_array=col_array[1:-2]
    else:
        print('Invalid option')
    indiv_dim = tile_grids(dims,
                           nblocks=[row_array,col_array],
                           indiv_grids=False)
    return dims,indiv_dim


def offset_tiling(W,nblocks=[10,10],offset_case=None):
    """
    Given a matrix W, which was split row and column wise
    given row_cut,col_cut, calculate three off-grid splits
    of the same matrix. Each offgrid will be only row-,
    only column-, and row and column-wise.
    Inputs:
    -------
    W:          np.array (d1 x d2 x T)
    r_offset:
    c_offset:
    row_cut:
    col_cut:
    Outputs:
    --------
    W_rs:       list
    W_cs:       list
    W_rcs:      list
    """
    dims=W.shape
    row_array,col_array = tile_grids(dims,nblocks)
    r_offset = vector_offset(row_array)
    c_offset = vector_offset(col_array)
    rc0, rc1 = (row_array[1:]-r_offset)[[0,-1]]
    cc0, cc1 = (col_array[1:]-c_offset)[[0,-1]]

    if offset_case is None:
        W_off = split_image_into_blocks(W,
                                        nblocks=nblocks)

    elif offset_case == 'r':
        W = W[rc0:rc1,:,:]
        W_off = split_image_into_blocks(W,
                                        nblocks=[row_array[1:-2],
                                                 col_array[1:-1]])
    elif offset_case == 'c':
        W = W[:,cc0:cc1,:]
        W_off = split_image_into_blocks(W,
                                        nblocks=[row_array[1:-1],
                                                 col_array[1:-2]])
    elif offset_case == 'rc':
        W = W[rc0:rc1,cc0:cc1,:]
        W_off = split_image_into_blocks(W,
                                        nblocks=[row_array[1:-2],
                                                 col_array[1:-2]])
    else:
        print('Invalid option')
        W_off = W
    return W_off, W.shape


def denoise_dx_tiles(W,
                     nblocks=[10,10],
                     dx=1,
                     maxlag=5,
                     confidence=0.99,
                     greedy=False,
                     fudge_factor=0.99,
                     mean_th_factor=1.15,
                     U_update=False,
                     min_rank=1,
                     stim_knots=None,
                     stim_delta=200):
    dims = W.shape
    start = time.time()
    W_ = split_image_into_blocks(W,nblocks=nblocks)
    print('----- split image into blocks %f' % (time.time() - start))
    dW_,rank_W_ = run_single(W_,
                             maxlag=maxlag,
                             confidence=confidence,
                             fudge_factor=fudge_factor,
                             mean_th_factor=mean_th_factor,
                             U_update=U_update,
                             min_rank=min_rank)
    dims_ = list(map(np.shape,dW_))
    start = time.time()
    dW_ = combine_blocks(dims,dW_,list_order='C')
    print('----- combine_blocks %f' % (time.time() - start))
    if dx ==1:
        return dW_, rank_W_
    W_ = None
    del W_
    dW_ = dW_.astype('float32')
    #get_process_memory()

    W_rs, drs = offset_tiling(W,
                             nblocks=nblocks,
                             offset_case='r')
    dW_rs,rank_W_rs = run_single(W_rs,
                                 maxlag=maxlag,
                                 confidence=confidence,
                                 fudge_factor=fudge_factor,
                                 mean_th_factor=mean_th_factor,
                                 U_update=U_update,
                                 min_rank=min_rank)
    dims_rs = list(map(np.shape,dW_rs))
    dW_rs = combine_blocks(drs,dW_rs,list_order='C')

    W_rs = None
    del W_rs
    dW_rs = dW_rs.astype('float32')
    #get_process_memory()

    W_cs, dcs = offset_tiling(W,
                     nblocks=nblocks,
                     offset_case='c')
    dW_cs,rank_W_cs = run_single(W_cs,
                                 maxlag=maxlag,
                                 confidence=confidence,
                                 fudge_factor=fudge_factor,
                                 mean_th_factor=mean_th_factor,
                                 U_update=U_update,
                                 min_rank=min_rank)
    dims_cs = list(map(np.shape,dW_cs))
    dW_cs = combine_blocks(dcs,dW_cs,list_order='C')

    W_cs = None
    del W_cs
    dW_cs = dW_cs.astype('float32')
    #get_process_memory();

    W_rcs, drcs = offset_tiling(W,
                      nblocks=nblocks,
                      offset_case='rc')
    dW_rcs,rank_W_rcs = run_single(W_rcs,
                             maxlag=maxlag,
                             confidence=confidence,
                             fudge_factor=fudge_factor,
                             mean_th_factor=mean_th_factor,
                             U_update=U_update,
                             min_rank=min_rank)
    dims_rcs = list(map(np.shape,dW_rcs))
    dW_rcs = combine_blocks(drcs,dW_rcs,list_order='C')

    W_rcs = None
    del W_rcs
    dW_rcs = dW_rcs.astype('float32')
    #get_process_memory()

    if False:
        return nblocks, dW_, dW_rs, dW_cs, dW_rcs, dims_, dims_rs, dims_cs, dims_rcs
    W_four = combine_4xd(nblocks,
                         dW_,
                         dW_rs,
                         dW_cs,
                         dW_rcs,
                         dims_,
                         dims_rs,
                         dims_cs,
                         dims_rcs)
    return W_four , [rank_W_,rank_W_rs,rank_W_cs,rank_W_rcs]


def combine_4xd(nblocks,dW_,dW_rs,dW_cs,dW_rcs,dims_,dims_rs,dims_cs,dims_rcs,plot_en=False):
    dims = dW_.shape
    row_array,col_array = tile_grids(dims,nblocks)
    r_offset = vector_offset(row_array)
    c_offset = vector_offset(col_array)
    r1, r2 = (row_array[1:]-r_offset)[[0,-1]]
    c1, c2 = (col_array[1:]-c_offset)[[0,-1]]
    drs     =   dW_rs.shape
    dcs     =   dW_cs.shape
    drcs    =   dW_rcs.shape
    # Get pyramid functions for each grid
    ak1 = np.zeros(dims[:2]).astype('float32')
    ak2 = np.zeros(dims[:2]).astype('float32')
    ak3 = np.zeros(dims[:2]).astype('float32')
    ak0 = pyramid_tiles(dims, dims_, list_order='C')
    ak1[r1:r2,:] = pyramid_tiles(drs, dims_rs, list_order='C')
    ak2[:,c1:c2] = pyramid_tiles(dcs, dims_cs, list_order='C')
    ak3[r1:r2,c1:c2] = pyramid_tiles(drcs, dims_rcs, list_order='C')
    # Force outer most border = 1
    ak0[[0,-1],:]=1
    ak0[:,[0,-1]]=1
    #return ak0,ak1,ak2,ak3,patches,W_rs,W_cs,W_rcs
    if False:
        return ak0,ak1,ak2,ak3
    W1 = np.zeros(dims).astype('float32')
    W2 = np.zeros(dims).astype('float32')
    W3 = np.zeros(dims).astype('float32')
    W1[r1:r2,:,:] = dW_rs
    W2[:,c1:c2,:] = dW_cs
    W3[r1:r2,c1:c2,:] = dW_rcs
    if plot_en:
        for ak_ in [ak0,ak1,ak2,ak3]:
            plt.figure(figsize=(10,10))
            plt.imshow(ak_[:,:])
            plt.show()
    if plot_en:
        plt.figure(figsize=(10,10))
        plt.imshow((ak0+ak1+ak2+ak3)[:,:])
        plt.colorbar()
    W_hat = ak0[:,:,np.newaxis]*dW_
    W_hat += ak1[:,:,np.newaxis]*W1
    W_hat += ak2[:,:,np.newaxis]*W2
    W_hat += ak3[:,:,np.newaxis]*W3
    W_hat /= (ak0+ak1+ak2+ak3)[:,:,np.newaxis]
    return W_hat


#salma
def run_single_dask(Y,
               maxlag=5,
               confidence=0.999,
               greedy=False,
               fudge_factor=0.99,
               mean_th_factor=1.15,
               U_update=False,
               min_rank=1,
               stim_knots=None,
               stim_delta=200):
    """
    Run denoiser in each movie in the list Y.
    Inputs:
    ------
    Y:      list (number_movies,)
            list of 3D movies, each of dimensions (d1,d2,T)
            Each element in the list can be of different size.
    Outputs:
    --------
    Yds:    list (number_movies,)
            list of denoised 3D movies, each of same dimensions
            as the corresponding input movie.input
    vtids:  list (number_movies,)
            rank or final number of components stored for each movie.
    ------
    """
    import dask

    start=time.time()

    result = []
    for patch in Y:
        result_batch = dask.delayed(gpca.denoise_patch)(patch,
                                                        maxlag=maxlag,
                                                        confidence=confidence,
                                                        greedy=greedy,
                                                        fudge_factor=fudge_factor,
                                                        mean_th_factor=mean_th_factor,
                                                        U_update=U_update,
                                                        min_rank=min_rank,
                                                        stim_knots=stim_knots,
                                                        stim_delta=stim_delta)


        result.append(result_batch)

    c_outs = dask.compute(*result)

    print('Total run time: %f'%(time.time()-start))
    Yds = [out_[0] for out_ in c_outs]
    vtids = [out_[1] for out_ in c_outs]
    vtids = np.asarray(vtids).astype('int')
    c_outs = None
    clear_variables(c_outs)
    #get_process_memory();
    return Yds,vtids



def run_single(Y,
               maxlag=5,
               confidence=0.999,
               greedy=False,
               fudge_factor=0.99,
               mean_th_factor=1.15,
               U_update=False,
               min_rank=1,
               stim_knots=None,
               stim_delta=200):
    """
    Run denoiser in each movie in the list Y.
    Inputs:
    ------
    Y:      list (number_movies,)
            list of 3D movies, each of dimensions (d1,d2,T)
            Each element in the list can be of different size.
    Outputs:
    --------
    Yds:    list (number_movies,)
            list of denoised 3D movies, each of same dimensions
            as the corresponding input movie.input
    vtids:  list (number_movies,)
            rank or final number of components stored for each movie.
    ------
    """
    #get_process_memory()
    cpu_count = multiprocessing.cpu_count()
    start=time.time()
    pool = multiprocessing.Pool(cpu_count)
    args=[[patch] for patch in Y]
    Y = None
    clear_variables(Y)
    #get_process_memory()
    # define params in function
    c_outs = pool.starmap(partial(gpca.denoise_patch,
                                  maxlag=maxlag,
                                  confidence=confidence,
                                  greedy=greedy,
                                  fudge_factor=fudge_factor,
                                  mean_th_factor=mean_th_factor,
                                  U_update=U_update,
                                  min_rank=min_rank,
                                  stim_knots=stim_knots,
                                  stim_delta=stim_delta),
                          args)
    pool.close()
    pool.join()
    print('Total run time: %f'%(time.time()-start))
    Yds = [out_[0] for out_ in c_outs]
    vtids = [out_[1] for out_ in c_outs]
    vtids = np.asarray(vtids).astype('int')
    c_outs = None
    clear_variables(c_outs)
    #get_process_memory();
    return Yds,vtids


def pyramid_matrix(dims,plot_en=False):
    """
    Compute a 2D pyramid function of size dims.
    Parameters:
    ----------
    dims:       tuple (d1,d2)
                size of pyramid function
    Outputs:
    -------
    a_k:        np.array (dims)
                 Pyramid function ranges [0,1],
                 where 0 indicates the boundary
                 and 1 the center.
    """
    a_k = np.zeros(dims[:2]).astype('float32')
    xc, yc = ceil(dims[0]/2),ceil(dims[1]/2)
    for ii in range(xc):
        for jj in range(yc):
            a_k[ii,jj]=max(dims)-min(ii,jj)
            a_k[-ii-1,-jj-1]=a_k[ii,jj]
    for ii in range(xc,dims[0]):
        for jj in range(yc):
            a_k[ii,jj]=a_k[ii,-jj-1]
    for ii in range(xc):
        for jj in range(yc,dims[1]):
            a_k[ii,jj]=a_k[-ii-1,jj]
    a_k = a_k.max() - a_k
    a_k /=a_k.max()
    if plot_en:
        plt.figure(figsize=(10,10))
        plt.imshow(a_k)
        plt.xticks(np.arange(dims[1]))
        plt.yticks(np.arange(dims[0]))
        plt.colorbar()
        plt.show()
    return a_k.astype('float32')


def pyramid_tiles(dims_rs,dims_,list_order='C',plot_en=False):
    """
    Calculate 2D array of size dims_rs,
    composed of pyramid matrices, each of which has the same
    dimensions as an element in W_rs.
    Inputs:
    -------
    dims_rs:    tuple (d1,d2)
                dimension of array
    W_rs:       list
                list of pacthes which indicate dimensions
                of each pyramid function
    list_order: order in which the
    Outputs:
    --------
    """
    a_ks = []
    for dim_ in dims_:
        a_k = pyramid_matrix(dim_)
        a_ks.append(a_k)
    # given W_rs and a_ks reconstruct array
    a_k = combine_blocks(dims_rs[:2],a_ks,dims_,list_order=list_order)
    if plot_en:
        plt.figure(figsize=(10,10))
        plt.imshow(a_k)
        plt.colorbar()
    return a_k


def cn_ranks(dim_block, ranks, dims, list_order='C'):
    """
    """
    Crank = np.zeros(shape=dims)*np.nan
    d1,d2  = Crank.shape
    i,j = 0,0
    for ii in range(0,len(ranks)):
        d1c , d2c  = dim_block[ii][:2]
        Crank[i:i+d1c,j:j+d2c].fill(int(ranks[ii]))
        if list_order=='F':
            i += d1c
            if i == d1:
                j += d2c
                i = 0
        else:
            j+= d2c
            if j == d2:
                i+= d1c
                j = 0
    return Crank


def combine_blocks(dimsM, Mc, dimsMc=None,
        list_order='C', array_order='F'):
    """
    Combine blocks given by compress_blocks

    Parameters:
    ----------
    dimsM:          tuple (d1,d2,T)
                    dimensions of original array
    Mc:             np.array or list
                    contains (padded) tiles from array.
    dimsMc:         np.array of tuples (d1,d2,T)
                    (original) dimensions of each tile in array
    list_order:     string {'F','C'}
                    determine order to reshape tiles in array
                    array order if dxT instead of d1xd2xT assumes always array_order='F'
                    NOTE: if dimsMC is NONE then MC must be a d1 x d2 x T array
    array_order:    string{'F','C'}
                    array order to concatenate tiles
                    if Mc is (dxT), the outputs is converted to (d1xd2xT)
    Outputs:
    --------
    M_all:          np.array (dimsM)
                    reconstruction of array from Mc
    """
    ndims = len(dimsM)
    if ndims ==3:
        d1, d2, T = dimsM
        Mall = np.zeros(shape=(d1, d2, T))*np.nan
    elif ndims ==2:
        d1,d2 = dimsM[:2]
        Mall = np.zeros(shape=(d1, d2))*np.nan
    if type(Mc)==list:
        k = len(Mc)
    elif type(Mc)==np.ndarray:
        k = Mc.shape[0]
    else:
        print('error= must be np.array or list')
    if dimsMc is None:
        dimsMc = np.asarray(list(map(np.shape,Mc)))
    i, j = 0, 0
    for ii, Mn in enumerate(Mc):
        # shape of current block
        d1c, d2c = dimsMc[ii][:2]
        if (np.isnan(Mn).any()):
            Mn = unpad(Mn)
        if Mn.ndim < 3 and ndims ==3:
            Mn = Mn.reshape((d1c, d2c)+(T,), order=array_order)
        if ndims ==3:
            Mall[i:i+d1c, j:j+d2c, :] = Mn
        elif ndims ==2:
            Mall[i:i+d1c, j:j+d2c] = Mn
        if list_order=='F':
            i += d1c
            if i == d1:
                j += d2c
                i = 0
        else:
            j += d2c
            if j == d2:
                i += d1c
                j = 0
    return Mall
