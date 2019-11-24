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
import globals
from globals import objFunc


globals.initGlobal()


# function definitions
def f(x, a, b, delta):
    fx = a ** 2 / ((b ** 2) + x) ** 2
    fx = np.sum(fx) - delta
    return fx


# def fp(x, a, b):
#     fpx = -2 * a ** 2 / ((b ** 2) + x) ** 3
#     fpx = np.sum(fpx)
#     return fpx


def objFunc(delta, finiteDiff=False):

    if finiteDiff:
        def fdiff(x, a, b):
            h = 1e-10
            result = (f(x+h, a, b, delta) - f(x, a, b, delta)) / h
            return result
        return fdiff
    else:
        def fp(x, a, b):
            fpx = -2 * a ** 2 / ((b ** 2) + x) ** 3
            fpx = np.sum(fpx)
            return fpx
        return fp


def g(xk, f1, fp1):
    xk = (-f1 / fp1) + xk
    return xk


# newton method implementation, no surprises here
# def NewtonRegFunc(x0, a, b, delta, f_tol, iterationLim, iterationStart, plotColor):
def NewtonRegFunc(x0, a, b, delta, f_tol, iterationLim, iterationStart, finiteDiff=False):

    fp = objFunc(delta, finiteDiff)

    iterations = iterationStart
    # iterations = 0

    xk = x0

    while iterations < iterationLim:

        f1 = f(xk, a, b, delta)

        if abs(f1) <= f_tol:
            break

        fp1 = fp(xk, a, b)

        # globals.plotFunc(iterations=iterations, f=f1, color=globals.global_plotColor[plotColor], yLogTruth=True, xLogTruth=False, markerType="^")

        xk = g(xk, f1, fp1)

        iterations += 1

    # print("For f(x) - delta = 0, x: %f" % xk)
    # print("In %d iterations\n" % iterations)

    return xk

