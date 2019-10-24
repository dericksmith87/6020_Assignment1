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
import matplotlib as mpl
from plotFunc import plotClustersFunc
import sklearn
from sklearn.cluster import KMeans
from prepData import prepDataFunc
from findClusters import clusterFunc
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance


fileString = "Spotted.jpg"
jpgData1 = plt.imread(fileString, format="jpg")
xLength = jpgData1.shape[0]
yLength = jpgData1.shape[1]

plt.imshow(jpgData1)
plt.title("No Clusters")
plt.show()


# time1 = time.clock()

# jpgLongArr1 = np.zeros((xLength, 3))
jpgLongArr1 = np.zeros((xLength * yLength, 3))

i = 0
k = 0
while i < xLength:

    j = 0
    # while j < 1:
    while j < yLength:
        jpgLongArr1[k, :] = jpgData1[i, j, :]
        j += 1
        k += 1

    i += 1

# xs = jpgLongArr1[:, 0]
# ys = jpgLongArr1[:, 1]
# zs = jpgLongArr1[:, 2]
# cs = jpgLongArr1 / 255
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.view_init(azim=170)
# i = 0
# # while i < xLength:
# while i < xLength*yLength:
#     c = mpl.colors.to_hex(cs[i, :])
#     ax.scatter(xs=xs[i], ys=ys[i], zs=zs[i], s=0.5, alpha=0.05, c=c)
#     i += 1
# plt.show()


# time2 = time.clock()

# print(time2 - time1)

print('images at various values of k: ')

numClusters = 1

while numClusters <= 64:

    skCentroids = sklearn.cluster.KMeans(n_clusters=numClusters, random_state=0).fit(jpgLongArr1)
    centroids = skCentroids.cluster_centers_

    jpgLongArr2 = copy.deepcopy(jpgLongArr1)

    i = 0
    # while i < xLength:
    while i < xLength*yLength:

        d_index = 0
        d_small = distance.euclidean(jpgLongArr2[i, :], centroids[d_index])

        j = 1
        while j < numClusters:
            d_temp = distance.euclidean(jpgLongArr2[i, :], centroids[j])

            if d_temp < d_small:
                d_small = d_temp
                d_index = j
            j += 1

        jpgLongArr2[i, :] = np.around(centroids[d_index])

        i += 1

    jpgData2 = copy.deepcopy(jpgData1)

    # print('plot with clusters: ')
    #
    #
    # xs = jpgLongArr1[:, 0]
    # ys = jpgLongArr1[:, 1]
    # zs = jpgLongArr1[:, 2]
    # cs = jpgLongArr2 / 255
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(azim=170)
    # i = 0
    # # while i < xLength:
    # while i < xLength*yLength:
    #     c = mpl.colors.to_hex(cs[i, :])
    #     ax.scatter(xs=xs[i], ys=ys[i], zs=zs[i], s=0.5, alpha=0.05, c=c)
    #     i += 1
    # plt.show()



    # print('return to image format: ')


    i = 0
    k = 0
    while i < xLength:

        j = 0
        while j < yLength:
            jpgData2[i, j, :] = jpgLongArr2[k, :]
            j += 1
            k += 1

        i += 1

    plt.imshow(jpgData2)
    plotTitle = "k = " + str(numClusters)
    plt.title(plotTitle)
    plt.show()

    numClusters += 1





#
# centroidsDf = pd.DataFrame(centroids[:, 0:2], columns=('x', 'y'))
# centroidsDf.to_csv(r'centroidsDf.csv', index=False)
# clusterDf.to_csv(r'clusterDf.csv', index=False)






print('hi')















print('hi')
