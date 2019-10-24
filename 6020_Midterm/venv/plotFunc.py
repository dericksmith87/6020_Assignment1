import pandas as pd
import numpy as np
import scipy as sp
import copy
import time
import numpy.random
import scipy.linalg
import scipy.ndimage
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plotClustersFunc(centroids, clusterDf, numClusters, titleString, iterationNum, repCount, kType):

    centroidsDf = pd.DataFrame(centroids[:, 0:2], columns=('x', 'y'))
    centroidsInitDf = pd.read_csv(r'centroidsInitDf.csv')
    plotColor = cm.rainbow(np.linspace(0, 1, numClusters))

    i = 0
    while i < numClusters:

        x = centroidsDf.loc[:, 'x']
        y = centroidsDf.loc[:, 'y']

        plt.scatter(x, y, alpha=0.8, c='c', edgecolors='none', s=50, marker="D")

        tempSubDf = clusterDf[clusterDf['c'] == i]
        x = tempSubDf.loc[:, 'x']
        y = tempSubDf.loc[:, 'y']

        plt.scatter(x, y, alpha=0.8, c=plotColor[i], edgecolors='none', s=15)

        i = i + 1

    x = centroidsInitDf.loc[:, 'x']
    y = centroidsInitDf.loc[:, 'y']
    plt.scatter(x, y, alpha=0.8, c='b', edgecolors='none', s=50, marker="^")

    titleString = titleString + ", i: " + str(iterationNum)
    plt.title(titleString)
    fileName = "vary_arrangement/" + kType + "_arrangement10_k" + str(numClusters) + "_rep" + str(repCount) + "_i" + str(iterationNum) + ".png"
    plt.savefig(fileName, dpi=72, bbox_inches='tight')
    plt.show()

