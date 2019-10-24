import pandas as pd
import numpy as np
import scipy as sp
import copy
import time
import numpy.random
import scipy.linalg
import scipy.ndimage
import sklearn
# import scikit-learn
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from plotFunc import plotClustersFunc
from sklearn.cluster import KMeans


dataPtDf = pd.read_csv(r'dataPtDf.csv')
dataPtArr = pd.DataFrame.__array__(dataPtDf)
clusterDf = pd.read_csv(r'clusterDf.csv')
numClusters = 4

skCentroids = sklearn.cluster.KMeans(n_clusters=numClusters, init='random', random_state=0).fit(dataPtArr)
centroids = skCentroids.cluster_centers_

plotClustersFunc(centroids, clusterDf, numClusters)



















