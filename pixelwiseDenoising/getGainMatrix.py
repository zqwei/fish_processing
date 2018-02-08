"""
Using spark to compute the gain matrix for image at different light intensity

One needs to login onto spark --


"""

# following is run while at server
import pathlib as pl
import thunder as td
from glob import glob


def get_stats(base_dir):
    from fish.util.fileio import read_image
    fnames = sorted(glob(base_dir + '/TM*'))
    ims = td.images.fromlist(fnames, accessor=read_image, engine=sc)

    im_means = ims.map(lambda v: v.mean(0)).toarray()
    im_vars = ims.map(lambda v: v.var(0)).toarray()
    ave_means = im_means.mean(0)
    ave_vars = im_vars.mean(0) + im_means.var(0)

    return ave_means, ave_vars

def get_gain_matrix_from_sever(base_dir, out_path):
    exp_names = [pl.Path(bd).parts[-1] for bd in base_dirs]
    results = [get_stats(bd) for bd in base_dirs]
    return


if __name__ == '__main__':
    base_dirs = glob('/nrs/ahrens/davis/data/spim/raw/20180206/*mW*')
    out_path = '/groups/ahrens/ahrenslab/davis/shared/for_zqiang/de'
    get_gain_matrix_from_sever(base_dir=base_dirs, out_path=out_path)
