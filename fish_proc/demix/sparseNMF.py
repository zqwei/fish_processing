import numpy as np

################################## code for sparse NMF, and simulating data ###################################
##################### vanilla nmf with random initialization with single penalty #########################
######### min|Y-UV|_2^2 + lambda*(|U|_1 + |V|_1) #####################

def vanilla_nmf_lasso(Yd, num_component, maxiter, tol, penalty_param, c=None):
    from sklearn import linear_model
    if Yd.min() < 0:
        Yd -= Yd.min(axis=2, keepdims=True);

    y0 = Yd.reshape(np.prod(Yd.shape[:2]),-1,order="F");
    if c is None:
        c = np.random.rand(y0.shape[1],num_component);
        c = c*np.sqrt(y0.mean()/num_component);

    clf_c = linear_model.Lasso(alpha=(penalty_param/(2*y0.shape[0])),positive=True,fit_intercept=False);
    clf_a = linear_model.Lasso(alpha=(penalty_param/(2*y0.shape[1])),positive=True,fit_intercept=True);
    res = np.zeros(maxiter);
    for iters in range(maxiter):
        temp = clf_a.fit(c, y0.T);
        a = temp.coef_;
        b = temp.intercept_;
        b = b.reshape(b.shape[0],1,order="F");
        c = clf_c.fit(a, y0-b).coef_;
        b = np.maximum(0, y0.mean(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));

        res[iters] = np.linalg.norm(y0 - np.matmul(a, c.T) - b,"fro")**2 + penalty_param*(abs(a).sum() + abs(c).sum());
        if iters > 0 and abs(res[iters] - res[iters-1])/res[iters-1] <= tol:
            break;
    if iters > 0:
        print(abs(res[iters] - res[iters-1])/res[iters-1]);

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

    corr_img_all_r = a.copy();
    for ii in range(a.shape[1]):
        corr_img_all_r[:,ii] = vcorrcoef2(y0, c[:,ii]);
    #corr_img_all_r = np.corrcoef(y0,c.T)[:y0.shape[0],y0.shape[0]:];
    corr_img_all_r = corr_img_all_r.reshape(Yd.shape[0],Yd.shape[1],-1,order="F");
    return {"a":a, "c":c, "b":b, "res":res, "corr_img_all_r":corr_img_all_r}


def nnls_L0(X, Yp, noise):
    """
    Nonnegative least square with L0 penalty, adapt from caiman
    It will basically call the scipy function with some tests
    we want to minimize :
    min|| Yp-W_lam*X||**2 <= noise
    with ||W_lam||_0  penalty
    and W_lam >0
    Parameters:
    ---------
        X: np.array
            the input parameter ((the regressor
        Y: np.array
            ((the regressand
    Returns:
    --------
        W_lam: np.array
            the learned weight matrices ((Models
    """
    from scipy.optimize import nnls
    W_lam, RSS = nnls(X, np.ravel(Yp))
    RSS = RSS * RSS
    if RSS > noise:  # hard noise constraint problem infeasible
        return W_lam

    print("hard noise constraint problem feasible!");
    while 1:
        eliminate = []
        for i in np.where(W_lam[:-1] > 0)[0]:  # W_lam[:-1] to skip background
            mask = W_lam > 0
            mask[i] = 0
            Wtmp, tmp = nnls(X * mask, np.ravel(Yp))
            if tmp * tmp < noise:
                eliminate.append([i, tmp])
        if eliminate == []:
            return W_lam
        else:
            W_lam[eliminate[np.argmin(np.array(eliminate)[:, 1])][0]] = 0


def vanilla_nmf_multi_lasso(y0, num_component, maxiter, tol, fudge_factor=1, c_penalize=True, penalty_param=1e-4):
    from ..utils.noise_estimator import noise_estimator
    from sklearn import linear_model
    sn = (noise_estimator(y0)**2)*y0.shape[1];
    c = np.random.rand(y0.shape[1],num_component);
    c = c*np.sqrt(y0.mean()/num_component);
    a = np.zeros([y0.shape[0],num_component]);
    res = np.zeros(maxiter);
    clf = linear_model.Lasso(alpha=penalty_param,positive=True,fit_intercept=False);
    for iters in range(maxiter):
        for ii in range(y0.shape[0]):
            a[ii,:] = nnls_L0(c, y0[[ii],:].T, fudge_factor * sn[ii]);
        if c_penalize:
            norma = (a**2).sum(axis=0);
            for jj in range(num_component):
                idx_ = np.setdiff1d(np.arange(num_component),ii);
                R_ = y0 - a[:,idx_].dot(c[:,idx_].T);
                V_ = (a[:,jj].T.dot(R_)/norma[jj]).reshape(1,y0.shape[1]);
                sv = (noise_estimator(V_)[0]**2)*y0.shape[1];
                c[:,jj] = nnls_L0(np.identity(y0.shape[1]), V_, fudge_factor * sv);
        else:
            #c = clf.fit(a, y0).coef_;
            c = np.maximum(0, np.matmul(np.matmul(np.linalg.inv(np.matmul(a.T,a)), a.T), y0)).T;
        res[iters] = np.linalg.norm(y0 - np.matmul(a, c.T),"fro");
        if iters > 0 and abs(res[iters] - res[iters-1])/res[iters-1] <= tol:
            break;
    if iters > 0:
        print(abs(res[iters] - res[iters-1])/res[iters-1]);
    return a, c, res


def sim_noise(dims, noise_source):
    np.random.seed(0);
    N = np.prod(dims);
    noise_source = noise_source.reshape(np.prod(noise_source.shape), order="F");
    random_indices = np.random.randint(0, noise_source.shape[0], size=N);
    noise_sim = noise_source[random_indices].reshape(dims,order="F");
    return noise_sim
