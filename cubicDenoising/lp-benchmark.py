from numpy import array, eye, hstack, ones, vstack, zeros
from pylab import random as pyrandom
from pylab import norm as pynorm
from sys import argv

# sample code to compare solvers"
# reference https://scaron.info/blog/linear-programming-in-python-with-pulp.html
available_solvers = []

try:
    import pulp
    pulp.COIN_CMD()
    available_solvers.append('pulp_coin')
except:
    pass

try:
    import pulp
    pulp.GLPK_CMD()
    available_solvers.append('pulp_glpk')
except:
    pass

try:
    import cvxopt
    import cvxopt.solvers
    available_solvers.append('cvxopt')
    cvxopt.solvers.options['show_progress'] = False  # disable output
except:
    pass

try:
    import cvxopt.glpk
    available_solvers.append('cvxopt_glpk')
    # disable GLPK output
    cvxopt.solvers.options['LPX_K_MSGLEV'] = 0         # old versions of cvxopt
    cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_OFF'  # works on cvxopt 1.1.7
except:
    pass

try:
    import cvxopt.mosek
    available_solvers.append('cvxopt_mosek')
except:
    pass


def pulp_solve_minmax(n, a, B, solver):
    x = pulp.LpVariable.dicts("x", range(n + 1), -42, +42)
    prob = pulp.LpProblem("Compute internal torques", pulp.LpMinimize)
    prob += pulp.lpSum([x[n]]), "Minimize_the_maximum"
    for i in range(n):
        label = "Max_constraint_%d" % i
        dot_B_x = pulp.lpSum([B[i][j] * x[j] for j in range(n)])
        condition = pulp.lpSum([x[n]]) >= a[i] + dot_B_x
        prob += condition, label
    prob.solve(solver)
    return array([v.value() for v in prob.variables()])


def cvxopt_solve_minmax(n, a, B, solver=None):
    # cvxopt objective format: c.T x
    c = hstack([zeros(n), [1]])

    # cvxopt constraint format: G * x - h <= 0
    # first,  a + B * x[0:n] <= x[n]
    G1 = zeros((n, n + 1))
    G1[0:n, 0:n] = B
    G1[:, n] = -ones(n)
    h1 = -a

    # then, x_min <= x <= x_max
    x_min = -42 * ones(n)
    x_max = +42 * ones(n)
    G2 = vstack([
        hstack([+eye(n), zeros((n, 1))]),
        hstack([-eye(n), zeros((n, 1))])])
    h2 = hstack([x_max, -x_min])

    c = cvxopt.matrix(c)
    G = cvxopt.matrix(vstack([G1, G2]))
    h = cvxopt.matrix(hstack([h1, h2]))
    sol = cvxopt.solvers.lp(c, G, h, solver=solver)
    return array(sol['x']).reshape((n + 1,))


def solve_random_minmax(n, solver_str):
    assert solver_str in available_solvers
    a, B = 2 * pyrandom(n) - 1., 2 * pyrandom((n, n)) - 1.
    if solver_str == 'cvxopt':
        return cvxopt_solve_minmax(n, a, B)
    elif solver_str == 'cvxopt_glpk':
        return cvxopt_solve_minmax(n, a, B, solver='glpk')
    elif solver_str == 'cvxopt_mosek':
        return cvxopt_solve_minmax(n, a, B, solver='mosek')
    elif solver_str == 'pulp_coin':
        return pulp_solve_minmax(n, a, B, pulp.COIN_CMD(msg=0))
    elif solver_str == 'pulp_glpk':
        return pulp_solve_minmax(n, a, B, pulp.GLPK_CMD(msg=0))


def unit_test(n):
    a, B = 2 * pyrandom(n) - 1., 2 * pyrandom((n, n)) - 1.
    sol1 = cvxopt_solve_minmax(n, a, B)
    sol2 = pulp_solve_minmax(n, a, B, pulp.GLPK_CMD(msg=0))
    if pynorm(sol1 - sol2) > 1e-3:
        print ("Solvers did not return the same solutions:")
        print ('cvxopt:', repr(sol1))
        print ('pulp_glpk:', repr(sol2))
        print ('n =', n)
        print ('a =', repr(a))
        print ('B =', repr(B))


if __name__ == '__main__':
    n = 5 if len(argv) < 2 else int(argv[1])
    unit_test(5)

    try:
        import IPython
        print ("Suggestions:\n")
        for i, s in enumerate(available_solvers):
            print ("%%timeit solve_random_minmax(%d, '%s')" % (n, s))
        print ("")
        IPython.embed()
    except:
        pass
