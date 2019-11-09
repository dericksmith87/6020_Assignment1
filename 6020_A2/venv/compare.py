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
from Newton_Like import NewtonLikeFunc
from BisectionMethod import findXzero, findFzero
import globals

globals.initGlobal()

def f(x, a, b, delta):
    fx = a ** 2 / ((b ** 2) + x) ** 2
    fx = np.sum(fx) - delta
    return fx


arrayNsizes = [10, 100, 1000, 10000]
arrayNsizes = np.asarray(arrayNsizes)
deltaSizes = [1e-5, 1e-8, 1e-11, 1e-14]
deltaSizes = np.asarray(deltaSizes)

# outer test loop to test four times at each array size
i = 0
while i < 4:

    # inner test loop to test various array sizes
    j = 0
    while j < 4:

        arrayN = arrayNsizes[j]
        a = np.random.uniform(-1000, 1000, arrayN)
        b = np.random.uniform(-1000, 1000, arrayN)

        delta = deltaSizes[i]
        f_tol = 1e-16
        iterations = 0
        iterationLim = 100

        x0 = findXzero(f, a, b, delta)

        print("Newton Regular:")
        NewtonRegFunc(x0, a, b, delta, f_tol, iterationLim, iterationStart=0, plotColor=0)

        print("Newton Like:")
        NewtonLikeFunc(x0, a, b, delta, f_tol, iterationLim)

        print("Bisection Method:")
        findFzero(f, a, b, delta, f_tol)

        plt.title("Iterations Per Method. Delta: %.1E . N: %d" % (delta, arrayN))
        plt.xlabel("Iterations")
        plt.ylabel("Log Residuals")

        # fileName = "figures/Delta%1.E N%d.png" % (delta, arrayN)
        # plt.savefig(fileName, dpi=72, bbox_inches='tight')

        plt.show()

        j += 1

    i += 1



