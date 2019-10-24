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


def prepDataFunc(fileString, fileType):
    sampleData = plt.imread(fileString, format=fileType)
    sampleShape = np.shape(sampleData)
    sampleHeight = sampleShape[0]
    sampleWidth = sampleShape[1]
    dataPtArr = np.empty(((sampleHeight * sampleWidth), 2))  # to store x and y coordinates

    i = 0
    x = 0
    while x < sampleHeight:

        y = 0
        while y < sampleWidth:

            if sampleData[x, y] > 100:
                dataPtArr[i, 0] = x
                dataPtArr[i, 1] = y
                i += 1

            y += 1

        x += 1

    # dataPtArr = dataPtArr[0:i - 1, ]
    dataPtArr = dataPtArr[0:i, ]
    dataPtDf = pd.DataFrame(dataPtArr, columns=('x', 'y'))
    dataPtDf.to_csv(r'dataPtDf.csv', index=False)
    return dataPtArr
