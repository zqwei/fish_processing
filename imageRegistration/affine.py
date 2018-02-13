from dipy.align.transforms import TranslationTransform2D, TranslationTransform3D
from dipy.align.transforms import RigidTransform2D, RigidTransform3D
from dipy.align.transforms import AffineTransform2D, AffineTransform3D
from dipy.align.imaffine import MutualInformationMetric, AffineRegistration

import dipy.io as dio

def estimate_affine(fixed, moving, trans_key = 'rigid'):
    assert len(moving.shape) == len(fixed.shape)
    if len(moving.shape) == 2:
        translation = TranslationTransform2D()
        rigid = RigidTransform2D()
        affine = AffineTransform2D()
    elif len(moving.shape) == 3:
        translation = TranslationTransform3D()
        rigid = RigidTransform3D()
        affine = AffineTransform3D()

    nbins = 32
    sampling_prop = .25
    metric = MutualInformationMetric(nbins, sampling_prop) #MI
    level_iters = [1000, 500, 250, 125]
    factors = [8, 4, 2, 1]
    sigmas = [3.0, 2.0, 1.0, 0.0]
    # method could be CG, BFGS, Newton-CG, dogleg or trust-ncg

    params0 = None

    affmap = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas,
                                factors=factors, method='L-BFGS-B', ss_sigma_factor=None,
                                options=None, verbosity=0)
    tx_tr = affmap.optimize(fixed, moving, translation, params0)
    if trans_key == 'translation':
        tx = tx_tr
    elif trans_key == 'rigid':
        tx = affmap.optimize(fixed, moving, rigid, params0, starting_affine = tx_tr.affine)
    elif trans_key == 'affine':
        tx = affmap.optimize(fixed, moving, rigid, params0, starting_affine = tx_tr.affine)
        tx = affmap.optimize(fixed, moving, affine, params0, starting_affine = tx.affine)

    return tx


def save_affine(fname, affine_):
    dio.save_pickle('affine', out_affine)


def transform_affine(fname, moving):
    out_ = dio.load_pickle(fname)
    return out_.transform(moving)
