"""
Using spark to compute the gain matrix for image at different light intensity

One needs to login onto spark --
1. request the nodes and ssh the user into the cluster when they are available

spark-janelia-lsf launch-in -n 10

2. start pyspark and a notebook server

spark-janelia-lsf start -b

3. -b is optional for notebook

"""

from glob import glob


def get_stats(base_dir):
    from fish.util.fileio import read_image
    import thunder as td
    fnames = sorted(glob(base_dir + '/TM*'))
    ims = td.images.fromlist(fnames, accessor=read_image, engine=sc)
    ims.cache()
    im_means = ims.map(lambda v: v.mean(0)).toarray()
    im_vars = ims.map(lambda v: v.var(0)).toarray()
    ave_means = im_means.mean(0)
    ave_vars = im_vars.mean(0) + im_means.var(0)
    ims.unpersist()
    return ave_means, ave_vars


def get_gain_matrix_from_sever(base_dir, out_path):
    import pathlib as pl
    import numpy as np
    exp_names = [pl.Path(bd).parts[-1] for bd in base_dirs]
    results = [get_stats(bd) for bd in base_dirs]
    for ind, exp in enumerate(exp_names):
        outfilename = out_path + exp
        np.savez(outfilename, results[ind][0], results[ind][1])


if __name__ == '__main__':
    base_dirs = glob('/nrs/ahrens/davis/data/spim/raw/20180206/*mW*')
    out_path = '/groups/ahrens/ahrenslab/davis/shared/for_zqiang/de'
    get_gain_matrix_from_sever(base_dir=base_dirs, out_path=out_path)
