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

# A2Q1a

# trid = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
nSize = 9
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


tridLU = sp.linalg.lu(trid)

# tridP = tridLU[0]
# tridL = tridLU[1]
tridU = tridLU[2]

# tridLLU = sp.linalg.lu(tridL)
# tridLL = tridLLU[1]
# tridLU = tridLLU[2]

# tridQR = np.linalg.qr(trid)
# tridQ = tridQR[0]
# tridR = tridQR[1]

# tridInv = np.linalg.inv(trid)
#
# tridInvLU = sp.linalg.lu(tridInv)
#
# # tridInvP = tridInvLU[0]
# tridInvL = tridInvLU[1]
# tridInvU = tridInvLU[2]

tridLDL = sp.linalg.ldl(trid)
tridL = tridLDL[0]
tridD = tridLDL[1]
tridLt = np.transpose(tridL)

# dotLDL = np.dot(tridL, tridD)
# dotLDL = np.dot(dotLDL, tridLt)

tridDsqrt = np.sqrt(tridD)
# tridDsqrtDot = np.dot(tridDsqrt, tridDsqrt)

tridLDsqrt = np.dot(tridL, tridDsqrt)
tridDsqrtLt = np.dot(tridDsqrt, tridLt)

tridLDsLt = np.dot(tridLDsqrt, tridDsqrtLt)



print('hi')






