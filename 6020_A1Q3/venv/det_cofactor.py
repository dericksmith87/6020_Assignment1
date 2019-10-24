import pandas as pd
import numpy as np
import scipy as sp
import copy
import scipy.linalg


def detCofactor(mx):
  n = mx[0,].size
  if n ==1: return mx[0,0]

  if n == 2: ##################### 4 flops
      return mx[0, 0] * mx[1, 1] - mx[1, 0] * mx[0, 1]

  mxDet = 0
  i = 0
  while i < n: ##################### loop runs n-1 of the previous matrix dimension
      tempMx1 = mx[0:i, 1:n]
      tempMx2 = mx[i + 1:n, 1:n]
      tempMx3 = np.concatenate((tempMx1, tempMx2), axis=0)
      mxDet = mxDet + (-1) ** i * mx[i, 0] * detCofactor(tempMx3)
      i = i + 1
  return mxDet






