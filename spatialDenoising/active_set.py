import numpy as np
import scipy
import scipy.linalg
import scipy.sparse as sparse
import primal_dual as pd

def primal_dual(y,
                lam,
                p=1,
                m=5,                           
                delta_s=.9,
                delta_e=1.1,
                maxiter=1e5,
                init_partition=None,
                init_theta=None,
                verbose=False):
    """
    """
    # Initialize Partition
    if init_partition is not None:
        P, N, A = init_partition
    elif init_theta is not None:
        P, N, A = primal_to_partition(init_theta)
    else:
        P, N, A = primal_to_partition(y)

    # Initialize dual and queue
    z = np.zeros(len(y)-2)
    Q = [0]*m

    # Main Iterations
    for itr in np.arange(1, maxiter+1).astype('int'):

        # Update Dual & Minimize Subspace
        z[P] = 1
        z[N] = -1
        theta, z = subspace_minimization(y, lam, z, A, np.sort(np.append(P, N)))

        # Locate Violations
        VP = np.argwhere(second_order_diff_subset(theta, P) < 0)
        nVP = len(VP)
        VN = np.argwhere(second_order_diff_subset(theta, N) > 0)
        nVN = len(VN)        
        VAP = np.argwhere(z[A] > 1)
        nVAP = len(VAP)
        VAN = np.argwhere(z[A] < -1)
        nVAN = len(VAN)
        V = np.append(np.append(P[VP], N[VN]),
                      np.append(A[VAP], A[VAN]))
        nV = len(V)

        # check termination criterion
        if nV == 0:
            # Proper Termination
            return theta, z, (P, N, A), True
        # Check Safeguard Queue For Cycles
        elif (nV > max(Q)) and (itr > m):
            p = max(delta_s * p, 1 / nV)
        elif (nV < min(Q)) and (itr > m):
            Q[itr % m] = nV
            p = max(delta_e * p, 1)
        else:
            Q[itr % m] = nV

        # Move top k violators
        k = int(round(p * nV))
        fitness = np.maximum(lam * np.abs(second_order_diff_subset(theta, V)),
                             np.abs(z[V]))
        move_idx = np.argsort(fitness * -1)[:k]  # only move largest k indices

        # Update violator partitions to only keep those to be moved
        VP = VP[move_idx[move_idx < nVP]]
        VN = VN[move_idx[np.logical_and(move_idx >= nVP,
                                           move_idx < nVP + nVN)]
                   - nVP]
        VAP = VAP[move_idx[np.logical_and(move_idx >= nVP + nVN,
                                             move_idx < nV - nVAN)]
                     - (nVP + nVN)]
        VAN = VAN[move_idx[move_idx >= nV - nVAN] - (nV - nVAN)]

        # Update partitions with chosen violators
        PVP = P[VP]
        P = np.sort(np.append(np.delete(P, VP), A[VAP]))
        NVN = N[VN]
        N = np.sort(np.append(np.delete(N, VN), A[VAN]))
        A = np.sort(np.append(np.delete(A, np.append(VAP, VAN)),
                              np.append(PVP, NVN)))
        print(itr)

    # Algorithm Did Not Terminate
    return theta, z, (P, N, A), False


def subspace_minimization(y, lam, z, A, I):
    """ subspce minimization step from ___ """
    if len(A) == 0:
        print("lA = 0")
        pass
    elif len(A) == 1:
        print("lA = 1")
        z[A] = second_order_diff_subset(y - lam * second_order_div_subset(z[I], I, len(y)), A) / (6* lam)
    else:
        z[A] = scipy.linalg.solve_banded(
            (2,2),
            second_order_diff_subset_gram(A).data,
            second_order_diff_subset(y - lam * second_order_div_subset(z[I], I, len(y)), A) / lam
        )
        
    theta = y - lam * pd.second_order_div(z)
    return theta, z


def second_order_diff_subset(theta, A):
    """ 
    Evaluate subset of rows (A) of the discrete 
    second order difference matrix against a vector theta....
    Assume that A is sorted
    """
    return (2 * theta[A+1]) - theta[A] - theta[A+2]


def second_order_div_subset(zI, I, n):
    """   """
    theta = np.zeros(n)
    theta[I] -= zI
    theta[I+1] += 2 * zI
    theta[I+2] -= zI
    return theta


def second_order_diff_subset_gram(A):
    """
    Computed D_A D_A^T for a set of indices A 
    Assume A sorted increasing
    """
    n = len(A)
    a = np.ones(n) * 6.0
    b = np.zeros(n-1)
    adj = A[1:] - A[:-1]
    b[adj == 1] = -4
    b[adj == 2] = 1
    c = np.zeros(n-2)
    c[A[2:] - A[:-2] == 2] = 1
    return sparse.diags([c, b, a, b, c], [-2, -1, 0, 1, 2]).transpose()
