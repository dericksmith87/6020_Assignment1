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
# from scipy import ldl

#A2Q1b

# trid = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
nSize = 7
trid = np.zeros((nSize, nSize))

trid[0, 0] = 2
trid[0, 1] = -1

i = 1
while i < nSize - 1:
    trid[i, i - 1] = -1
    trid[i, i] = 2
    trid[i, i + 1] = -1
    i += 1

trid[nSize - 1, nSize - 1] = 2
trid[nSize - 1, nSize - 2] = -1


# trid = [[2, -1, 0, 0, 0], [-1, 2, -1, 0, 0], [0, -1, 2, -1, 0], [0, 0, -1, 2, -1], [0, 0, 0, -1, 2]]
# trid = np.asarray(trid)

tridSVD = np.linalg.svd(trid)
tridEIG = np.linalg.eig(trid)

tridSVDu = tridSVD[0]
tridSVDsigmas = tridSVD[1]
# tridSVDsigmas = np.transpose(tridSVD[1])
tridSVDv = tridSVD[2]

tridEigval = tridEIG[0]
tridEigvec = tridEIG[1]

TridEigvec = np.dot(trid, tridEigvec[:, 0])
EigvalEigvec = np.dot(tridEigval[0], tridEigvec[:, 0])

idMx = np.identity(nSize)

Lambdas = np.dot(idMx, tridEigval)

# composeTrid = np.dot(tridEigvec, )

print(tridEigvec)
print('hi')






