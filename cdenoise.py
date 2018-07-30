


def low_rank_svd(img, rank_k):
    from sklearn.utils.extmath import randomized_svd
    dx, dy, dt = img.shape
    vec_img = img.reshape(dx*dy, dt)
    U, s, Vt = randomized_svd(vec_img, n_components=rank_k, n_iter=7, random_state=None)
    return U.dot(np.diag(s).dot(Vt)).reshape((dx, dy, dt))

def pad_index(arr, overlap_step, max_value):
    if arr[0]>overlap_step:
        arr = np.append(arr, arr[0]+np.arange(-overlap_step,0))
    if arr.max()<max_value-overlap_step:
        arr = np.append(arr, arr.max()+np.arange(1,overlap_step))
    return np.sort(arr.flatten())

def split_to_blocks(nsize, nblocks, overlap_step):
    assert len(nsize) == len(overlap_step), print("lengths not matched")
    arr = []
    for nsize_, nblock_, overlap_ in zip(nsize, nblocks, overlap_step):
        arr_ = np.array_split(range(nsize_), nblock_)
        arr_ = [pad_index(_, overlap_, nsize_) for _ in arr_]
        arr.append(arr_)
    return arr

def get_blocks_from_index_arr(imgStack, arrs):
    blocks = []
    for z_arr in arrs[0]:
        for x_arr in arrs[1]:
            for y_arr in arrs[2]:
                block_ = imgStack[z_arr.min():z_arr.max()+1, x_arr.min():x_arr.max()+1, y_arr.min():y_arr.max()+1, :]
                blocks.append([block_, z_arr, x_arr, y_arr])
    return blocks

def run_single(blocks, maxlag=5, confidence=0.999, greedy=False, fudge_factor=0.99, mean_th_factor=1.15,
               U_update=False, min_rank=1, stim_knots=None, stim_delta=200):

    from . import greedyPCA as gpca
    import time
    import multiprocessing
    from functools import partial

    func = partial(gpca.denoise_patch, maxlag=maxlag, confidence=confidence, greedy=greedy,
                                  fudge_factor=fudge_factor, mean_th_factor=mean_th_factor, U_update=U_update,
                                  min_rank=min_rank, stim_knots=stim_knots, stim_delta=stim_delta)

    start=time.time()
    cpu_count = max(1, multiprocessing.cpu_count()-2)
    args=[patch[0] for patch in blocks]
    start=time.time()
    pool = multiprocessing.Pool(cpu_count)
    print('Running %d blocks in %d cpus'%(len(blocks), cpu_count))#if verbose else 0
    # define params in function
    c_outs = pool.map(func, args)
    pool.close()
    pool.join()
    Yds = [out_[0] for out_ in c_outs]
    vtids = [out_[1] for out_ in c_outs]
    vtids = np.asarray(vtids).astype('int')
    print('Blocks(=%d) run time: %f'%(len(Y),time.time()-start))
    return Yds, vtids

def combine_blocks(block_data, block_size, block_corrs):
    block_mat = np.zeros(block_size)
    block_count = block_mat.copy().astype(np.int)
    for ndata, ncorr in zip(block_data, block_corrs):
        _, z_arr, x_arr, y_arr = ncorr
        block_mat[z_arr.min():z_arr.max()+1, x_arr.min():x_arr.max()+1, y_arr.min():y_arr.max()+1, :] += ndata
        block_count[z_arr.min():z_arr.max()+1, x_arr.min():x_arr.max()+1, y_arr.min():y_arr.max()+1, :] += 1
    return block_mat, block_count
