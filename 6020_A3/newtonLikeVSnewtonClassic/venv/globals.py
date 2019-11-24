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


def objFunc(f, h):

    def nestFunc(x):
        result = (f(x+h) - f(x)) / h
        return result

    return nestFunc


def initGlobal():
    global global_plotColor
    # global_plotColor = cm.rainbow(np.linspace(0, 1, 3))
    global_plotColor = ["red", "blue", "green"]







