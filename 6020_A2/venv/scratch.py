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

# hollowMatrix = [[0, 1, 2], [3, 0, 4], [5, 6, 0]]
#
# print(hollowMatrix)
# print(np.linalg.det(hollowMatrix))

mSize = 5
# randL = np.zeros((mSize, mSize))
# randU = np.zeros((mSize, mSize))

# norm1 = np.linalg.norm(randM, 1)
# norm2 = np.linalg.norm(randM, 2)
# print(norm1)
# print(norm2)
# print(norm1/norm2)
# l = 0
# while l < 100:
#     randM = np.random.uniform(-100, 100, (mSize, mSize))
#     i = 0
#     while i < mSize:
#         randM[i, i] = 0
#         randL[i, 0:i] = randM[i, 0:i]
#         randU[i, i+1:mSize] = randM[i, i+1:mSize]
#         i += 1


# detM = np.linalg.det(randM)
# detL = np.linalg.det(randL)
# detU = np.linalg.det(randU)


# print(detM)
# print(detL)
# print(detU)
#
# l += 1

# randM = np.random.uniform(-0.999999, 0.999999, (mSize, mSize))
M = [[1, 0.2, 0.25, 0.35], [0.1, 1, 0.4, 0.15], [0.2, 0.22, 1, 0.33], [0.3, 0.3, 0.27, 1]]
M = np.asarray(M)

# Mt = np.transpose(M)


Mt = np.transpose(M)

MMt = np.dot(M, Mt)

MMt_1 = copy.deepcopy(MMt)

MMt_1[:, 0] = MMt[:, 0] / MMt[0, 0]
MMt_1[:, 1] = MMt[:, 1] / MMt[1, 1]
MMt_1[:, 2] = MMt[:, 2] / MMt[2, 2]
MMt_1[:, 3] = MMt[:, 3] / MMt[3, 3]
#
# MM2 = np.dot(MM1_1, MM1_1)
# MM1_1 = np.dot(MM1_1, MM1_1)
# MM1_1 = np.dot(MM1_1, MM1_1)
# MM1_1 = np.dot(MM1_1, MM1_1)

print("hi")

# print(np.sum(M[:,0]))
# print(np.sum(M[:,1]))
# print(np.sum(M[:,2]))
# print(np.sum(M[:,3]))


# i = 0
# while i < mSize:
#     randM[i, i] = 0
#     # randL[i, 0:i] = randM[i, 0:i]
#     # randU[i, i+1:mSize] = randM[i, i+1:mSize]
#     i += 1

# randMt = np.transpose(randM)
# MMt = np.dot(randM, randMt)
# MtM = np.dot(randMt, randM)
# MM = np.dot(randM, randM)

# MtMDotMMt = MtM * MMt

# print(MtM)
# print(MMt)
