import pandas as pd
import numpy as np
import scipy as sp
import copy
import time
import numpy.random
import scipy.linalg
import scipy.ndimage
import sklearn
import time
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Newton import NewtonRegFunc
import globals


globals.initGlobal()


def f(x, a, b, delta):
    fx = a ** 2 / ((b ** 2) + x) ** 2
    fx = np.sum(fx) - delta
    return fx


def fp(x, a, b):
    fpx = -2 * a ** 2 / ((b ** 2) + x) ** 3
    fpx = np.sum(fpx)
    return fpx


# rational model
def g(xkr, a, b, delta):

    magicNumber = 42    # I can't explain it, it just works. It's magic.

    fk = f(xkr, a, b, delta)
    fpk = fp(xkr, a, b)

    xk1 = -fk / fpk + xkr
    fk1 = f(xk1, a, b, delta)

    # Formulas for A and B derived from g(x) explained in written portion
    B = -fk / fpk - xkr
    A = fk * (B + xkr)

    # calculate next iteration of x by solving g(x_k+1) - f(x_k)= 0
    xkr = A / (fk1 / magicNumber) - B

    return xkr


# basic iteration converging algorithm with helper function
# similar in many ways to Newton method implementation
def NewtonLikeFunc(x0, a, b, delta, f_tol, iterationLim):

    NewtonSummoned = False
    iterations = 0

    xkr = x0
    x_previous = xkr + 10

    while iterations < iterationLim:

        f1 = f(xkr, a, b, delta)

        globals.plotFunc(iterations=iterations, f=f1, color=globals.global_plotColor[1], yLogTruth=True, xLogTruth=False, markerType="s")

        # if the g(x) model stalls out, Newton picks up the slack
        if f1 < 0 or abs(xkr - x_previous) < 10 and f1 < 1:
            print("Newton summoned at iteration %d" % iterations)
            NewtonRegFunc(x_previous, a, b, delta, f_tol, iterationLim, iterationStart=iterations+1, plotColor=1)
            NewtonSummoned = True
            break

        if np.abs(f1) <= f_tol:
            break

        x_previous = xkr

        # solving for next x iteration using rational model g(x)
        xkr = g(xkr, a, b, delta)

        iterations += 1

    if not NewtonSummoned:
        print("For f(x) - delta = 0, x: %f" % xkr)
        print("In %d iterations\n" % iterations)
