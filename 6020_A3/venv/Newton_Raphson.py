# Newton-Raphson iteration. Programmed in lecture 12.
import numpy as np
from copy import copy
from globals import objFunc


def Newton_Raphson(f, fp, x0, eps_f, eps_x, N, finiteDiff=False):

    x = copy(x0)
    r = f(x)
    if finiteDiff:
        # h = 1e-8
        h = 1e-10
        # if delta_min < h:
        #     h = delta_min
        fp = objFunc(f, h, r, x0)

    # In: function handles for f and its Jacobian, intial guess, tolerance for residual and error, max nr of iterations.
    diag = []
    # x = copy(x0)
    for i in range(N):
        r = f(x)
        J = fp(x)
        # linalg.solve uses LUP decomposition
        dx = -np.linalg.solve(J, r)
        x += dx
        # 2-norm used for residual and error
        res = np.linalg.norm(r, 2)
        err = np.linalg.norm(dx, 2)
        print('%d %e %e' % (i, res, err))
        diag.append([i, res, err])
        if res < eps_f and err < eps_x:
            print('Convergence after %d iterations!' % N)
            return x, np.asarray(diag)
    print('No convergence after %d iterations!' % N)
    return x, np.asarray(diag)
