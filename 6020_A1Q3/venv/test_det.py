import pandas as pd
import numpy as np
import scipy as sp
import copy
import scipy.linalg
import time
import matplotlib.pyplot as plt
from det_LUP import detLUP
from det_cofactor import detCofactor



########## Controls and Constants ##########
setInit = 2 # starting mxA dim 2x2
setLim = 12 # max mxA dim (setLim - 1)x(setLim - 1)
setTotal = setLim - setLim
repLim = 50 # number of times each matrix dimension test is repeated
dataColNum = 11
numMethods = 3
methodNames = ('LUP', 'CoFactor', 'LUP (numpy)')

runTestLoops = False
runAnalysis = False
runRegression = True

########## Controls and Constants##########






########## Test Loops Commence ##########

if runTestLoops:

    mxData = np.zeros(((setTotal * repLim), dataColNum))
    dfData = pd.DataFrame(mxData, columns=['t_lup', 't_co', 't_py', 'det_lup', 'det_co', 'det_py', 'absE_lup_to_py',
                                           'relE_lup_to_py', 'absE_co_to_py', 'relE_co_to_py', 'setNum'])

    i = setInit # i := dim of mxA
    k = 0
    while i < setLim:

        j = 0
        while j < repLim:

            mxA = np.random.random((i, i))

            t1 = time.clock()

            dLUP = detLUP(mxA)
            dfData.loc[k, 'det_lup'] = dLUP

            t2 = time.clock()

            dCo = detCofactor(mxA)
            dfData.loc[k, 'det_co'] = dCo

            t3 = time.clock()

            dPy = np.linalg.det(mxA)
            dfData.loc[k, 'det_py'] = dPy

            t4 = time.clock()

            dfData.loc[k, 't_lup'] = t2 - t1
            dfData.loc[k, 't_co'] = t3 - t2
            dfData.loc[k, 't_py'] = t4 - t3

            dDiff = dLUP - dPy
            dRel = dDiff / dPy
            dfData.loc[k, 'absE_lup_to_py'] = np.absolute(dDiff)
            dfData.loc[k, 'relE_lup_to_py'] = np.absolute(dRel)

            dDiff = dCo - dPy
            dRel = dDiff / dPy
            dfData.loc[k, 'absE_co_to_py'] = np.absolute(dDiff)
            dfData.loc[k, 'relE_co_to_py'] = np.absolute(dRel)

            dfData.loc[k, 'setNum'] = i

            j = j + 1
            k = k + 1

        i = i + 1

    dfData.to_csv(r'dfData.csv', index=False)

########## Test Loops Complete ##########


########## Analysis Commence ##########

if runAnalysis:

    if ~runTestLoops:
        dfData = pd.read_csv(r'dfData.csv')

    AnalysisColNum = 6
    mxAnalysis = np.zeros((setTotal, AnalysisColNum)) #
    dfAnalysis = pd.DataFrame(mxAnalysis, columns=('dim', 'mu_time', 's_time', 'mu_relE', 's_relE', 'method'))
    # method "smarty" variable := {lup = 0, co = 1, py = 2}

    m = 0  # method
    p = 0
    while m < numMethods:  # py is measuring stick, not included

        i = setInit

        while i < setLim:

            dfSetSub = dfData[dfData['setNum'] == i]

            if m < numMethods - 1:
                dfAnalysis.loc[p,'mu_relE'] = np.mean(dfSetSub.iloc[:, 7+2*m]) # position 7 == relE_lup_to_py, 9 == relE_co_to_py
                dfAnalysis.loc[p,'s_relE'] = np.std(dfSetSub.iloc[:, 7+2*m])
            dfAnalysis.loc[p,'mu_time'] = np.mean(dfSetSub.iloc[:, 0+m]) # position 1 == t_lup, 2 == t_co, 3 == t_py
            dfAnalysis.loc[p,'s_time'] = np.std(dfSetSub.iloc[:, 0+m])
            dfAnalysis.loc[p,'method'] = m
            dfAnalysis.loc[p,'dim'] = i

            i = i + 1
            p = p + 1

        m = m + 1

    dfAnalysis.to_csv(r'dfAnalysis.csv', index=False)

########## Analysis Complete ##########


########## Regression Commence ##########

if runRegression:

    if ~runTestLoops:
        dfData = pd.read_csv(r'dfData.csv')

    if ~runAnalysis:
        dfAnalysis = pd.read_csv(r'dfAnalysis.csv')

    i = 0
    while i < numMethods: # regression of py time

        dfMethodSub = dfAnalysis[dfAnalysis['method'] == i]

        titleName = methodNames[i] + ' Average Time per Dimension n with Log(Time) Scale'
        plot1 = dfMethodSub.plot(kind='scatter', x='dim', y='mu_time', logy=True, title=titleName)

        plt.show()

        titleName = methodNames[i] + ' Average Error per Dimension n with Log(Error) Scale'
        plot1 = dfMethodSub.plot(kind='scatter', x='dim', y='mu_relE', logy=True, title=titleName)

        plt.show()

        i = i + 1


########## Regression Complete ##########









