from dipy.align.imaffine import MutualInformationMetric, AffineRegistration
from dipy.align.transforms import TranslationTransform2D, TranslationTransform3D
from dipy.align.transforms import RigidTransform2D, RigidTransform3D
from dipy.align.transforms import AffineTransform2D, AffineTransform3D
from dipy.align.imaffine import AffineMap
import dipy.io as dio
import numpy as np

class ImAffine:
    def __init__(self):
        self.nbins = 32
        self.sampling_prop = None #.25
        self.level_iters = [1000, 500, 250, 125]
        self.factors = [8, 4, 2, 1]
        self.sigmas = [3.0, 2.0, 1.0, 0.0]
        self.ss_sigma_factor=None
        self.verbosity = 0
        self.tx_mat = None
        self.params0 = None
        
        self.options = {'maxcor': 10, 'ftol': 1e-7, 'gtol': 1e-5, 'eps': 1e-8, 'maxiter': 1000, 'disp':True}
        # ftol: The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
        # gtol: The iteration will stop when max{|proj g_i | i = 1, ..., n} <= gtol where pg_i is the i-th component of the projected gradient.
        # eps: Step size used for numerical approximation of the jacobian.
        # disp: Set to True to print convergence messages.

        self.method='L-BFGS-B'
        self.metric = MutualInformationMetric(self.nbins, self.sampling_prop)
        self.affmap = AffineRegistration(metric=self.metric, level_iters=self.level_iters, sigmas=self.sigmas,
                                    factors=self.factors, method=self.method, ss_sigma_factor=self.ss_sigma_factor,
                                    options=self.options, verbosity=self.verbosity)
        self.update_map = True

    def estimate_translation2d(self, fixed, moving):
        assert len(moving.shape) == len(fixed.shape)
        trans = TranslationTransform2D()
        if self.update_map:
            self.metric = MutualInformationMetric(self.nbins, self.sampling_prop)
            self.affmap = AffineRegistration(metric=self.metric, level_iters=self.level_iters, sigmas=self.sigmas,
                                        factors=self.factors, method=self.method, ss_sigma_factor=self.ss_sigma_factor,
                                        options=self.options, verbosity=self.verbosity)
        return self.affmap.optimize(fixed, moving, trans, self.params0)

    def estimate_rigid2d(self, fixed, moving, tx_tr=None):
        assert len(moving.shape) == len(fixed.shape)
        trans = RigidTransform2D()
        if self.update_map:
            self.metric = MutualInformationMetric(self.nbins, self.sampling_prop)
            self.affmap = AffineRegistration(metric=self.metric, level_iters=self.level_iters, sigmas=self.sigmas,
                                        factors=self.factors, method=self.method, ss_sigma_factor=self.ss_sigma_factor,
                                        options=self.options, verbosity=self.verbosity)
        if tx_tr is None:
            self.update_map = False
            tx_tr = self.estimate_translation2d(fixed, moving)
            self.update_map = True
        if isinstance(tx_tr, AffineMap):
            tx_tr = tx_tr.affine
        return self.affmap.optimize(fixed, moving, trans, self.params0, starting_affine=tx_tr)


    def estimate_affine2d(self, fixed, moving, tx_tr=None):
        assert len(moving.shape) == len(fixed.shape)
        trans = AffineTransform3D()
        if self.update_map:
            self.metric = MutualInformationMetric(self.nbins, self.sampling_prop)
            self.affmap = AffineRegistration(metric=self.metric, level_iters=self.level_iters, sigmas=self.sigmas,
                                        factors=self.factors, method=self.method, ss_sigma_factor=self.ss_sigma_factor,
                                        options=self.options, verbosity=self.verbosity)
        if tx_tr is None:
            self.update_map = False
            tx_tr = self.estimate_rigid2d(fixed, moving)
            self.update_map = True
        if isinstance(tx_tr, AffineMap):
            tx_tr = tx_tr.affine
        return self.affmap.optimize(fixed, moving, trans, self.params0, starting_affine = tx_tr)


    def estimate_translation3d(self, fixed, moving):
        assert len(moving.shape) == len(fixed.shape)
        tx_tr = self.estimate_translation2d(fixed.mean(axis=0), moving.mean(axis=0))
        tx_tr = tx_tr.affine
        tmp = np.eye(4)
        tmp[1:, 1:] = tx_tr
        trans = TranslationTransform3D()
        if self.update_map:
            self.metric = MutualInformationMetric(self.nbins, self.sampling_prop)
            self.affmap = AffineRegistration(metric=self.metric, level_iters=self.level_iters, sigmas=self.sigmas,
                                        factors=self.factors, method=self.method, ss_sigma_factor=self.ss_sigma_factor,
                                        options=self.options, verbosity=self.verbosity)
        return self.affmap.optimize(fixed, moving, trans, self.params0, starting_affine=tmp)

    def estimate_rigid3d(self, fixed, moving, tx_tr=None):
        assert len(moving.shape) == len(fixed.shape)
        trans = RigidTransform3D()
        if self.update_map:
            self.metric = MutualInformationMetric(self.nbins, self.sampling_prop)
            self.affmap = AffineRegistration(metric=self.metric, level_iters=self.level_iters, sigmas=self.sigmas,
                                        factors=self.factors, method=self.method, ss_sigma_factor=self.ss_sigma_factor,
                                        options=self.options, verbosity=self.verbosity)
        if tx_tr is None:
            tmp = self.estimate_rigid2d(fixed.mean(axis=0), moving.mean(axis=0))
            tmp = tmp.affine
            tx_tr = np.eye(4)
            tx_tr[1:, 1:] = tmp
        if isinstance(tx_tr, AffineMap):
            tx_tr = tx_tr.affine
        return self.affmap.optimize(fixed, moving, trans, self.params0, starting_affine=tx_tr)

    def estimate_rigidxy(self, fixed, moving, tx_tr=None):
        assert len(moving.shape) == len(fixed.shape)
        trans = TranslationTransform3D()
        if self.update_map:
            self.metric = MutualInformationMetric(self.nbins, self.sampling_prop)
            self.affmap = AffineRegistration(metric=self.metric, level_iters=self.level_iters, sigmas=self.sigmas,
                                        factors=self.factors, method=self.method, ss_sigma_factor=self.ss_sigma_factor,
                                        options=self.options, verbosity=self.verbosity)
        if tx_tr is None:
            tmp = self.estimate_rigid2d(fixed.mean(axis=0), moving.mean(axis=0))
            tmp = tmp.affine
            tx_tr = np.eye(4)
            tx_tr[1:, 1:] = tmp
        if isinstance(tx_tr, AffineMap):
            tx_tr = tx_tr.affine

        trans2d = AffineMap(tx_tr, domain_grid_shape=fixed.shape, codomain_grid_shape=moving.shape)
        moving_ = trans2d.transform(fixed)
        transz = self.affmap.optimize(moving_, moving, trans, self.params0)
        print(transz.affine)
        tx_tr[0, 3] = transz.affine[0, 3]
        return AffineMap(tx_tr, domain_grid_shape=fixed.shape, codomain_grid_shape=moving.shape)

    def estimate_rigid_projz(self, fixed, moving, tx_tr=None):
        # this returns a 3d rotation matrix
        assert len(moving.shape) == len(fixed.shape)
        if tx_tr is None:
            tmp = self.estimate_rigid2d(fixed.mean(axis=0), moving.mean(axis=0))
            tmp = tmp.affine
            tx_tr = np.eye(4)
            tx_tr[1:, 1:] = tmp
        else:
            if isinstance(tx_tr, AffineMap):
                tx_tr = tx_tr.affine
            if tx_tr.shape[0] == 3:
                tmp = np.eye(4)
                tmp[1:, 1:] = tx_tr
                tx_tr = tmp
            tmp = self.estimate_rigid2d(fixed.mean(axis=0), moving.mean(axis=0),tx_tr=tx_tr)
            tmp = tmp.affine
            tx_tr = np.eye(4)
            tx_tr[1:, 1:] = tmp
        return AffineMap(tx_tr, domain_grid_shape=fixed.shape, codomain_grid_shape=moving.shape)


    def estimate_affine3d(self, fixed, moving, tx_tr=None):
        assert len(moving.shape) == len(fixed.shape)
        trans = AffineTransform3D()
        if self.update_map:
            self.metric = MutualInformationMetric(self.nbins, self.sampling_prop)
            self.affmap = AffineRegistration(metric=self.metric, level_iters=self.level_iters, sigmas=self.sigmas,
                                        factors=self.factors, method=self.method, ss_sigma_factor=self.ss_sigma_factor,
                                        options=self.options, verbosity=self.verbosity)
        if tx_tr is None:
            tmp = self.estimate_affine2d(fixed.mean(axis=0), moving.mean(axis=0))
            tmp = tmp.affine
            tx_tr = np.eye(4)
            tx_tr[1:, 1:] = tmp
        if isinstance(tx_tr, AffineMap):
            tx_tr = tx_tr.affine()
        return self.affmap.optimize(fixed, moving, trans, self.params0, starting_affine = tx_tr)

    def save_affine(fname, affine_):
        dio.save_pickle('affine', affine_)

    def transform_affine(fname, fix):
        out_ = dio.load_pickle(fname)
        return out_.transform(fix)
