# optimize of Newton-hook. Programmed in lectures 13 -- ...
import numpy as np
from copy import copy
from Newton import Newton
from Newton_Like import NewtonLikeFunc
from Newton_Classic import NewtonRegFunc
from BisectionMethod import findXzero
from globals import objFunc

def fr(x, a, b, delta):
    fx = a ** 2 / ((b ** 2) + x) ** 2
    fx = np.sum(fx) - delta
    return fx


def optimize(x, dx, f, df, delta, finiteDiff=False):
    f_tol = 1e-12
    # tol_x = 1e-12

    if finiteDiff:
        h = 1e-12
        # if delta_min < h:
        #     h = delta_min
        df = objFunc(f, h)

    iterationLim = 100

    N = np.shape(x)[0]
    J = df(x)
    U, S, Vt = np.linalg.svd(J)
    print('min sig=%e' % (np.min(S)))
    r = f(x)
    g = U.T @ r
    a = np.asarray(S)
    a = np.square(a) #?
    b = a
    x0 = findXzero(fr, a, b, delta)
    deltaSqrd = delta ** 2
    # lm = NewtonRegFunc(x0, a, b, deltaSqrd, f_tol, iterationLim, iterationStart=0, finiteDiff=True)
    lm = NewtonRegFunc(x0, a, b, deltaSqrd, f_tol, iterationLim, iterationStart=0, finiteDiff=False)
    # lm = NewtonLikeFunc(x0, a, b, deltaSqrd, f_tol, iterationLim, iterationStart=0)
    w = np.zeros((N, 1))
    for i in range(N):
        w[i] = -S[i] * g[i] / (S[i] ** 2 + lm)
    dx = Vt.T @ w
    return dx
