import pandas as pd
import numpy as np
import scipy as sp
import copy
import time
import numpy.random
import scipy.linalg
import scipy.ndimage
import scipy.stats as stats
import sklearn
import time
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

normA = 0
normYxT = 0
normDiff = 0
normDiffSum = 0
normSqrResSum = 0
normDiffArr = np.zeros(200*200)
l = 0
count = 0
while l < 200:
    i = 0
    arrayN = i + 1
    while i < 200:
        x = np.random.uniform(-10000, 10000, arrayN)
        xT= np.transpose(x)
        y = np.random.uniform(-10000, 10000, arrayN)
        A = np.outer(y, xT)

        normA = np.linalg.norm(A, 2)
        normYxT = np.linalg.norm(y, 2) * np.linalg.norm(x, 2)
        normDiff = normA - normYxT
        normDiffArr[count] = normDiff
        normDiffSum += normDiff
        normSqrResSum += normDiff ** 2

        arrayN += 1
        count += 1
        i += 1


    # count += i
    l += 1

# normAmu = normA / i
# normYxTmu = normYxT / i

normDiffMu = normDiffSum / i
normSd = np.sqrt(normSqrResSum / (count - 1))

testResult = stats.ttest_1samp(a=normDiffArr, popmean=0)
print(testResult)
print("After %d computations, the average difference in the two norms was %2.E with a standard deviation of %2.E." % (count, normDiffMu, normSd))
