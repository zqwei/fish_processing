from dipy.align.imaffine import MutualInformationMetric, AffineRegistration

import dipy.io as dio

def estimate_translation(fixed, moving, nbins=32, sampling_prop=.25, level_iters = [1000, 500, 250, 125], factors = [8, 4, 2, 1], sigmas = [3.0, 2.0, 1.0, 0.0], ss_sigma_factor=None):
    assert len(moving.shape) == len(fixed.shape)
    from dipy.align.transforms import TranslationTransform2D, TranslationTransform3D
    if len(moving.shape) == 2:
        trans = TranslationTransform2D()
    elif len(moving.shape) == 3:
        trans = TranslationTransform3D()

    metric = MutualInformationMetric(nbins, sampling_prop) #MI
    params0 = None
    affmap = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas,
                                factors=factors, method='L-BFGS-B', ss_sigma_factor=ss_sigma_factor,
                                options=None, verbosity=0)

    return affmap.optimize(fixed, moving, trans, params0)

def estimate_rigid(fixed, moving, nbins=32, sampling_prop=.25, level_iters = [1000, 500, 250, 125], factors = [8, 4, 2, 1], sigmas = [3.0, 2.0, 1.0, 0.0], ss_sigma_factor=None):
    assert len(moving.shape) == len(fixed.shape)
    from dipy.align.transforms import RigidTransform2D, RigidTransform3D
    if len(moving.shape) == 2:
        trans = RigidTransform2D()
    elif len(moving.shape) == 3:
        trans = RigidTransform3D()

    metric = MutualInformationMetric(nbins, sampling_prop) #MI
    params0 = None
    affmap = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas,
                                factors=factors, method='L-BFGS-B', ss_sigma_factor=ss_sigma_factor,
                                options=None, verbosity=0)

    tx_tr = estimate_translation(fixed, moving, nbins=nbins,
                                 sampling_prop=sampling_prop,level_iters=level_iters,
                                 factors = factors, sigmas = sigmas, ss_sigma_factor=ss_sigma_factor)

    return affmap.optimize(fixed, moving, trans, params0, starting_affine = tx_tr.affine)


def estimate_affine(fixed, moving, nbins=32, sampling_prop=.25, level_iters = [1000, 500, 250, 125], factors = [8, 4, 2, 1], sigmas = [3.0, 2.0, 1.0, 0.0], ss_sigma_factor=None, tx_tr=None):
    assert len(moving.shape) == len(fixed.shape)
    from dipy.align.transforms import AffineTransform2D, AffineTransform3D
    if len(moving.shape) == 2:
        trans = AffineTransform2D()
    elif len(moving.shape) == 3:
        trans = AffineTransform3D()

    metric = MutualInformationMetric(nbins, sampling_prop) #MI
    # method could be CG, BFGS, Newton-CG, dogleg or trust-ncg
    params0 = None
    affmap = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas,
                                factors=factors, method='L-BFGS-B', ss_sigma_factor=ss_sigma_factor,
                                options=None, verbosity=0)
    if tx_tr is None:
        tx_tr = estimate_rigid(fixed, moving, nbins=nbins,
                                     sampling_prop=sampling_prop,level_iters=level_iters,
                                     factors = factors, sigmas = sigmas, ss_sigma_factor=ss_sigma_factor)

    return affmap.optimize(fixed, moving, trans, params0, starting_affine = tx_tr.affine)


def save_affine(fname, affine_):
    dio.save_pickle('affine', out_affine)


def transform_affine(fname, moving):
    out_ = dio.load_pickle(fname)
    return out_.transform(moving)
