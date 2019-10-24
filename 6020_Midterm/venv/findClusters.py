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
from plotFunc import plotClustersFunc
import sklearn
from sklearn.cluster import KMeans


def clusterFunc(centroidsInit, numClusters, dataPtArr, centroids, repCount):

    time1 = time.clock()

    dataPtHeight = dataPtArr.shape[0]
    centroidsInitDf = pd.DataFrame(centroidsInit, columns=('x', 'y'))
    centroidsInitDf.to_csv(r'centroidsInitDf.csv', index=False)

    emptyMx = np.empty((dataPtHeight, 4))
    clusterDf = pd.DataFrame(emptyMx, columns=('x', 'y', 'd', 'c'))
    clusterDf.iloc[:, 0:2] = dataPtArr

    # first pass, initialize distance column
    clusterDf.loc[:, 'd'] = clusterDf.max()[0] ** 2 + clusterDf.max()[1] ** 2

    # subsequent passes
    iterationNum = 1
    while np.sum(centroids[:, 0:2] - centroids[:, 2:4]) != 0:

        # assign nearest centroids
        i = 0
        while i < dataPtHeight:
            d_temp = (clusterDf.loc[i, 'x'] - centroids[0, 0]) ** 2 + (clusterDf.loc[i, 'y'] - centroids[0, 1]) ** 2
            # d_temp = abs(clusterDf.loc[i, 'x'] - centroids[0, 0]) + abs(clusterDf.loc[i, 'y'] - centroids[0, 1])
            clusterDf.loc[i, 'd'] = d_temp
            clusterDf.loc[i, 'c'] = 0
            i = i + 1

        i = 0
        while i < dataPtHeight:

            j = 1
            while j < numClusters:

                d_temp = (clusterDf.loc[i, 'x'] - centroids[j, 0]) ** 2 + (clusterDf.loc[i, 'y'] - centroids[j, 1]) ** 2
                # d_temp = abs(clusterDf.loc[i, 'x'] - centroids[j, 0]) + abs(clusterDf.loc[i, 'y'] - centroids[j, 1])
                d_current = clusterDf.loc[i, 'd']
                if d_temp < d_current:
                    clusterDf.loc[i, 'd'] = d_temp
                    clusterDf.loc[i, 'c'] = j

                j = j + 1

            i = i + 1

        # update new centroids
        centroids[:, 2:4] = centroids[:, 0:2]  # store previous centroids

        j = 0
        while j < numClusters:
            subDf = clusterDf[clusterDf['c'] == j]
            avgSubX = np.sum(subDf.loc[:, 'x'])
            avgSubY = np.sum(subDf.loc[:, 'y'])
            if subDf.shape[0] == 0:
                print("Error: empty centroid cluster")

            avgSubX = avgSubX / subDf.shape[0]
            avgSubY = avgSubY / subDf.shape[0]

            # avgSubY = np.sum(subDf.loc[:, 'y']) / subDf.shape[0]
            centroids[j, 0] = avgSubX
            centroids[j, 1] = avgSubY

            j = j + 1


        iterationNum += 1

    inertia = np.int(np.sum(clusterDf.loc[:, 'd']))
    time2 = time.clock()
    timeTotal = np.around(time2 - time1, 4)

    titleString = "k-means. k: " + str(numClusters) + ", inertia: " + str(inertia) + ", time: " + str(timeTotal)
    plotClustersFunc(centroids, clusterDf, numClusters, titleString, iterationNum - 1, repCount, "km")

    centroidsDf = pd.DataFrame(centroids[:, 0:2], columns=('x', 'y'))

    return centroidsDf



