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
import matplotlib.cm as cm


def plotFunc(iterations, f, color, yLogTruth, xLogTruth, markerType):
    # plt.scatter(iterations, f, alpha=0.8, c=color, edgecolors='none', s=45, marker=markerType)
    # if yLogTruth:
    #     plt.yscale('log')
    # if xLogTruth:
    #     plt.xscale('log')
    return 0


def objFunc(f, h, r, x0):

    numX = np.size(x0)
    numY = np.size(r)

    J = np.zeros((numX, numY))

    def finiteDiffFunc(x):

        xCount = 0
        while xCount < numX:

            yCount = 0
            while yCount < numY:

                xh = copy.deepcopy(x)
                xh[yCount] = xh[yCount] + h
                # xh[xCount] = xh[xCount] + h
                yh = f(xh)
                yx = f(x)
                J[xCount, yCount] = (yh[xCount] - yx[xCount]) / h

                yCount += 1

            xCount += 1

        return J

    return finiteDiffFunc


def initGlobal():
    global global_plotColor
    # global_plotColor = cm.rainbow(np.linspace(0, 1, 3))
    global_plotColor = ["red", "blue", "green"]







