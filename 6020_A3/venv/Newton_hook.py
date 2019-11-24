# Newton-hook iteration. Programmed in lectures 13 -- ...
import numpy as np
from copy import copy
from optimize import optimize
from try_step import try_step
from globals import objFunc


def Newton_hook(x0, f, df, delta, N_max, tol_f, tol_x, alpha, beta, delta_min, finiteDiff=False):

    x = copy(x0)
    r = f(x)
    if finiteDiff:
        # h = 1e-8
        h = 1e-10
        if delta_min < h:
            h = delta_min
        df = objFunc(f, h, r, x0)


    k = 0
    # x = copy(x0)
    converged = 0
    diag = []
    while not converged:
        if k > N_max or delta < delta_min:
            break
        # compute the Newton-Raphson update step
        r = f(x)
        res = np.linalg.norm(r, 2)
        J = df(x)
        # linalg.solve uses LUP decomposition
        dx = -np.linalg.solve(J, r)
        size = np.linalg.norm(dx, 2)

        diag.append([k, res, size, delta])
        print('Step %d, delta=%e, res=%e size of NR step=%e ...' % (k, delta, res, size))

        if size < delta:
            accept = try_step(x, dx, f, res)  # set z <- x+dx, compute f(z), return accept = 1 iff |f(z)|< res
            if accept:
                # the Newton-Raphson step reduced the residual, accept it
                x += dx
                k += 1
                print('Accepted full NR step.')
            else:
                # the Newton-Raphson step increased the residual, reject it and decrease delta
                delta *= alpha
                print('Rejected full NR step, resetting delta=%e.' % delta)
        else:
            # dx = optimize(x, dx, f, df, delta, finiteDiff=True)
            dx = optimize(x, dx, f, df, delta)
            accept = try_step(x, dx, f, res)
            if accept:
                # the update step on the edge of the trust region gives a smaller residual, accept and increase delta
                x += dx
                delta *= beta
                k += 1
                print('Accepted constrained minimization step, resetting delta=%e.' % delta)
            else:
                # the update step on the edge of the trust region gives a larger residual, reject and reduce delta
                delta *= alpha
                print('Rejected constrained minimization step, resetting delta=%e.' % delta)

        r = f(x)
        res = np.linalg.norm(r, 2)
        err = np.linalg.norm(dx, 2)
        if res < tol_f and err < tol_x:
            converged = 1
            print('Convergence after %d iterations' % k)
            break

    if converged == 0:
        print('No convergence after %d iterations, delta = %e.' % (k, delta))

    return x, np.asarray(diag)
