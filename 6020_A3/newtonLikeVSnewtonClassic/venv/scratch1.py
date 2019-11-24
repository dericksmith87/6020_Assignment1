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
from Newton import NewtonRegFunc
import globals


M = np.random.uniform(0, 100, (10, 10))

# Msvd = np.linalg.svd(M)
Msig = np.linalg.svd(M)[1]
# Msig = Msvd[1]

Meig = np.linalg.eig(M)[0]
# MeigVal = Meig[1]

MeighSqrd = Meig ** 2

print(Msig)

print(Meig)

print("hi")


