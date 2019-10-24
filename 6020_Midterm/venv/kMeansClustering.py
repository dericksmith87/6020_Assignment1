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
from plotFunc import plotClustersFunc
from sklearn.cluster import KMeans
from prepData import prepDataFunc
from findClusters import clusterFunc

numClusters = 4  # k clusters
iterationLim = 10
iterationCount = 0
prepData = True
clusterData = True
# plotClusters = False

# Data Prep Commence

if prepData:

    dataPtArr = prepDataFunc("SampleData/Arrangement10.jpg", "jpg")

# Data Prep Complete


# Cluster Data Commence
while iterationCount < iterationLim:



    # if clusterData:

        # if ~prepData:
        #     dataPtDf = pd.read_csv(r'dataPtDf.csv')
        #     dataPtArr = pd.DataFrame.__array__(dataPtDf)

    dataPtHeight = dataPtArr.shape[0]
    centroidsInit = np.random.choice(dataPtHeight, size=numClusters, replace=False)
    # centroidsInit = np.random.randint(low=0, high=dataPtHeight, size=numClusters, dtype='I')
    centroids = np.empty((numClusters, 4))  # first two columns: current centroid, second two columns: previous centroid
    centroids[:, 0:2] = dataPtArr[centroidsInit, :]
    centroidsInit = centroids[:, 0:2]
    centroidsInitDf = pd.DataFrame(centroidsInit, columns=('x', 'y'))
    centroidsInitDf.to_csv(r'centroidsInitDf.csv', index=False)

    centroidsDf = clusterFunc(centroidsInit, numClusters, dataPtArr, centroids, iterationCount)

    emptyMx = np.empty((dataPtHeight, 4))
    clusterDf = pd.DataFrame(emptyMx, columns=('x', 'y', 'd', 'c'))
    clusterDf.iloc[:, 0:2] = dataPtArr
    clusterDf.iloc[:, 0:2] = dataPtArr

    time1 = time.clock()

    skCentroids = sklearn.cluster.KMeans(n_clusters=numClusters, random_state=0).fit(dataPtArr)

    inertia = np.int(skCentroids.inertia_)
    time2 = time.clock()
    timeTotal = np.around(time2 - time1, 4)



    # assign nearest centroids
    i = 0
    while i < dataPtHeight:
        d_temp = (clusterDf.loc[i, 'x'] - centroids[0, 0]) ** 2 + (clusterDf.loc[i, 'y'] - centroids[0, 1]) ** 2
        clusterDf.loc[i, 'd'] = d_temp
        clusterDf.loc[i, 'c'] = 0
        i = i + 1

    i = 0
    while i < dataPtHeight:

        j = 1
        while j < numClusters:

            d_temp = (clusterDf.loc[i, 'x'] - centroids[j, 0]) ** 2 + (clusterDf.loc[i, 'y'] - centroids[j, 1]) ** 2
            d_current = clusterDf.loc[i, 'd']
            if d_temp < d_current:
                clusterDf.loc[i, 'd'] = d_temp
                clusterDf.loc[i, 'c'] = j

            j = j + 1

        i = i + 1


    iterationNum = skCentroids.n_iter_

    centroidsInit = np.zeros((numClusters, 2))
    centroidsInitDf = pd.DataFrame(centroidsInit, columns=('x', 'y'))
    centroidsInitDf.to_csv(r'centroidsInitDf.csv', index=False)

    centroids = skCentroids.cluster_centers_

    titleString = "k-means++. k: " + str(numClusters) + ", inertia: " + str(inertia) + ", time: " + str(timeTotal)
    plotClustersFunc(centroids, clusterDf, numClusters, titleString, iterationNum, iterationCount, "kmpp")

    iterationCount += 1

# Cluster Data Complete
