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

globals.initGlobal()

# A helper function that uses 2^n to find when f(x_k+1)<0
# It then return x_k as x_0 to be used in all methods
# def findXzero(f, a, b, delta):
#
#     x0 = 1
#     x_previous = 0
#     i = 0
#     while i < 100:
#
#         f0 = f(x0, a, b, delta)
#
#         if f0 < 0:
#             x0 = (x_previous + x0) / 2
#
#         elif f0 < 1:
#             return x0
#
#         elif f0 > 1:
#             x0 *= 2
#
#         x_previous = x0
#
#         i+=1


def findXzero(f, a, b, delta):

    x0 = 1
    x_previous = 0
    i = 0
    while i < 100:

        f0 = f(x0, a, b, delta)
        #
        # if 0 < f0 < 1:
        #     print("Found x0 in %d iterations" % i)
        #     return x0

        if f0 < 0:
            # print("Found x0 in %d iterations" % i)
            i += 1
            return [x_previous, i]

        x_previous = x0
        x0 *= 10

        i += 1



# a simplified version of bisection that takes advantage of the single root of f(x)
# the same principles still apply
def findFzero(f, a, b, delta, f_tol):

    x0 = 10
    x_previous = 10
    iterations = 0

    i = 0
    while i < 1000:

        f0 = f(x0, a, b, delta)

        globals.plotFunc(iterations=iterations, f=f0, color=globals.global_plotColor[2], yLogTruth=True, xLogTruth=False, markerType="o")

        if f0 < 0:
            break
        elif f0 > 0:
            x_previous = x0
            x0 *= 10

        i += 1
        iterations += 1

    xa = x_previous
    xb = x0
    xm = (xa + xb) / 2

    i = 0
    while i < 10000:

        fm = f(xm, a, b, delta)

        globals.plotFunc(iterations=iterations, f=fm, color=globals.global_plotColor[2], yLogTruth=True, xLogTruth=False, markerType="o")

        if 0 <= abs(fm) <= f_tol:
            break
        elif fm < 0:
            xb = xm
        elif fm > 0:
            xa = xm

        xm = (xa + xb) / 2

        i += 1
        iterations += 1

    print("For f(x) - delta = 0, x: %f" % xm)
    print("In %d iterations\n" % iterations)

