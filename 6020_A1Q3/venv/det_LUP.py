import pandas as pd
import numpy as np
import scipy as sp
import copy
import scipy.linalg


def detLUP(mxA):
    dimA = mxA[0,].size
    mxP = np.identity(dimA)
    mxL = np.identity(dimA)
    mxU = copy.deepcopy(mxA)
    pSwaps = 0
    # print(mxU)

    i = 0  # row
    while i < dimA - 1: ##################### loop runs n-1 times

        # find largest element i to n
        bigI = i
        bigRow = i
        bigNum = mxU[bigI, i]
        bigI = bigI + 1
        while bigI < dimA: ##################### loop runs n-1 times with 1 flop
            if bigNum < mxU[bigI, i]:
                bigNum = mxU[bigI, i]
                bigRow = bigI
            bigI = bigI + 1
        # find largest element i to n

        # swap
        if bigRow != i: ##################### 2*3*n flops
            mxU[i,] = mxU[i,] + mxU[bigRow,]
            mxU[bigRow,] = mxU[i,] - mxU[bigRow,]
            mxU[i,] = mxU[i,] - mxU[bigRow,]

            mxP[i,] = mxP[i,] + mxP[bigRow,]
            mxP[bigRow,] = mxP[i,] - mxP[bigRow,]
            mxP[i,] = mxP[i,] - mxP[bigRow,]

            pSwaps = pSwaps + 1
        # swap

        # from each row below ith row, subtract by row (leading element/biggest)*big row
        l = i + 1
        while l < dimA: ##################### loop runs sum(n-l) from l=1 to l=(n-1) times with (1+n) flops per run

            mxL[l, i] = mxU[l, i] / mxU[i, i]
            mxU[l,] = mxU[l,] - mxL[l, i] * mxU[i,]

            l = l + 1

        i = i + 1
        # from each row below ith row, subtract by row (leading element/biggest)*big row

    i = 0
    mxAdet = (-1) ** pSwaps ##################### 1 flop
    while i < dimA: ##################### loop runs n times with 1 flop per run
        mxAdet = mxAdet * mxU[i, i]
        i = i + 1

    # print(mxAdet)
    return mxAdet

