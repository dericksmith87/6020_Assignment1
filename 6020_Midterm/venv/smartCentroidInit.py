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


############################# TRIED TO USE BINNING ##########################################
############################# DID NOT WORK AS HELPER FOR CLUSTERING ##########################################
############################# COULD WORK AS HELPER TO TYPES OF REGRESSION POTENTIALLY ##########################################

numClusters = 4  # k clusters
binMultiplier = 1
dataPtDf = pd.read_csv(r'dataPtDf.csv')


xRange = dataPtDf.max()[0] - dataPtDf.min()[0]
yRange = dataPtDf.max()[1] - dataPtDf.min()[1]

xBin = xRange / (binMultiplier*numClusters)
yBin = yRange / (binMultiplier*numClusters)

dataBinDf = copy.deepcopy(dataPtDf)
# dataBinDf.loc[:, 'x'] = (dataBinDf.loc[:, 'x'] // xBin) * xBin
# dataBinDf.loc[:, 'y'] = (dataBinDf.loc[:, 'y'] // yBin) * yBin
dataBinDf.loc[:, 'x'] = dataBinDf.loc[:, 'x'] // xBin
dataBinDf.loc[:, 'y'] = dataBinDf.loc[:, 'y'] // yBin

# x = dataBinDf.loc[:, 'x']
# y = dataBinDf.loc[:, 'y']
# plt.scatter(x, y, alpha=0.2, c='c', edgecolors='none', s=1000//binMultiplier, marker="o")
# plt.show()

numBins = (numClusters + 1) ** 2
countArr = np.zeros((numBins, 3))
# i = 0
# while i < numClusters**2:
#     tempArr[i, 0] = i // numClusters
#     tempArr[i, 1] = i % numClusters
#     i = i + 1


# dataFreqsDf = pd.DataFrame(tempArr, columns=('x', 'y', 'count'))
# nDataPts = dataPtDf.shape[0]

i = 0
while i < dataBinDf.shape[0]:
    xTemp = dataBinDf.loc[i, 'x']
    yTemp = dataBinDf.loc[i, 'y']
    freqIndex = int(xTemp * numClusters + yTemp)
    countArr[freqIndex, 0] = xTemp
    countArr[freqIndex, 1] = yTemp
    countArr[freqIndex, 2] += 1
    i += 1

x = countArr[:, 0]
y = countArr[:, 1]
count = countArr[:, 2]
plt.scatter(x, y, alpha=0.3, c='c', edgecolors='none', s=1000*count, marker="o")
plt.show()

binCentroidDf = pd.DataFrame(countArr, columns=('x', 'y', 'count'))
binCentroidDf.loc[:, 'x'] *= xBin
binCentroidDf.loc[:, 'y'] *= yBin
binCentroidDf = binCentroidDf[binCentroidDf['count'] != 0]
binCentroidDf.to_csv(r'binCentroidDf.csv', index=False)


# while numClusters < 20:
#     xRange = dataPtDf.max()[0] - dataPtDf.min()[0]
#     yRange = dataPtDf.max()[1] - dataPtDf.min()[1]
#
#     xBin = xRange / (binMultiplier*numClusters)
#     yBin = yRange / (binMultiplier*numClusters)
#
#     dataBinDf = copy.deepcopy(dataPtDf)
#     # dataBinDf.loc[:, 'x'] = (dataBinDf.loc[:, 'x'] // xBin) * xBin
#     # dataBinDf.loc[:, 'y'] = (dataBinDf.loc[:, 'y'] // yBin) * yBin
#     dataBinDf.loc[:, 'x'] = dataBinDf.loc[:, 'x'] // xBin
#     dataBinDf.loc[:, 'y'] = dataBinDf.loc[:, 'y'] // yBin
#
#     # x = dataBinDf.loc[:, 'x']
#     # y = dataBinDf.loc[:, 'y']
#     # plt.scatter(x, y, alpha=0.2, c='c', edgecolors='none', s=1000//binMultiplier, marker="o")
#     # plt.show()
#
#     numBins = (numClusters + 1) ** 2
#     countArr = np.zeros((numBins, 3))
#     # i = 0
#     # while i < numClusters**2:
#     #     tempArr[i, 0] = i // numClusters
#     #     tempArr[i, 1] = i % numClusters
#     #     i = i + 1
#
#
#     # dataFreqsDf = pd.DataFrame(tempArr, columns=('x', 'y', 'count'))
#     # nDataPts = dataPtDf.shape[0]
#
#     i = 0
#     while i < dataBinDf.shape[0]:
#         xTemp = dataBinDf.loc[i, 'x']
#         yTemp = dataBinDf.loc[i, 'y']
#         freqIndex = int(xTemp * numClusters + yTemp)
#         countArr[freqIndex, 0] = xTemp
#         countArr[freqIndex, 1] = yTemp
#         countArr[freqIndex, 2] += 1
#         i += 1
#
#     x = countArr[:, 0]
#     y = countArr[:, 1]
#     count = countArr[:, 2]
#     plt.scatter(x, y, alpha=0.3, c='c', edgecolors='none', s=1000*count, marker="o")
#     plt.show()
#
#     numClusters += 1
# while binMultiplier <= 100:
#     xBin = xRange / (binMultiplier*numClusters)
#     yBin = yRange / (binMultiplier*numClusters)
#
#     dataBinDf = copy.deepcopy(dataPtDf)
#     dataBinDf.loc[:, 'x'] = (dataBinDf.loc[:, 'x'] // xBin) * xBin
#     dataBinDf.loc[:, 'y'] = (dataBinDf.loc[:, 'y'] // yBin) * yBin
#
#     x = dataBinDf.loc[:, 'x']
#     y = dataBinDf.loc[:, 'y']
#     plt.scatter(x, y, alpha=0.2, c='c', edgecolors='none', s=1000//binMultiplier, marker="o")
#     plt.show()
#     binMultiplier = binMultiplier + 1














print('hi')




# bin each dimension with some proportion of k
# each dimension receives k
# create centroids based on all tuple permutations
# merge centroids with smallest distance until k centroids remain





















