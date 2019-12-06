import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import copy
import time
import pylab
import numpy.random
import scipy.linalg
import scipy.ndimage
import scipy.stats
import sklearn
import time
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans


def yHatFunc(x, betas):
    betas = betas[::-1]
    deg = betas.shape[0]
    numObs = x.shape[0]
    yHat = np.zeros(numObs)
    i = 0
    while i < deg:
        yHat = yHat + betas[i] * x ** i
        i += 1
    return yHat


def lineFunc(x, betas):
    x_domain = np.linspace(start=min(x), stop=max(x), num=100)
    yHats = yHatFunc(x_domain, betas)
    return x_domain, yHats


def yFunc(x0, x1):
    y = 0.5 * x0 ** 2 + 5 * x0 * np.sqrt(x1)
    numObs = x0.shape[0]
    epsilon = np.random.normal(0, 1, numObs)
    epsilon = np.abs(epsilon)
    y += epsilon
    return y


###############
# Create Data #
x0 = np.linspace(start=0, stop=10, num=100)  # known variable
x1 = copy.deepcopy(x0)                      # hidden variable
np.random.shuffle(x1)
y = yFunc(x0, x1)
###############


##############
# Tranform y #
plt.scatter(x0, y)
plt.show()

ySqrt = np.sqrt(y)
plt.scatter(x0, ySqrt)
plt.show()

yLg = np.log(y)
plt.scatter(x0, yLg)
plt.show()

print("hi")

y = ySqrt
##############



#######################
# Degree zero polyfit #
#######################
fit0 = np.polyfit(x=x0, y=y, deg=0)
yHat0 = yHatFunc(x=x0, betas=fit0)
res0 = y - yHat0
yHatLine0 = lineFunc(x=x0, betas=fit0)

    # Plot fit
plt.plot(yHatLine0[0], yHatLine0[1], color='r')
plt.scatter(x=x0, y=y)
plt.title("Fit 0: y and yHat")
plt.show()

    # Plot residuals
zeroLine0 = lineFunc(x=x0, betas=np.zeros(1))
plt.plot(zeroLine0[0], zeroLine0[1], color='r')
plt.scatter(x=x0, y=res0)
plt.title("Fit 0: residuals per x")
plt.show()

    # Histogram Residuals
plt.hist(res0)
plt.title("Fit 0: Residual Frequencies")
plt.show()

    # QQ Plot Residuals
zRes0 = (res0 - np.mean(res0)) / np.std(res0)
sm.qqplot(zRes0, line='45')
pylab.title("Fit 0: Q-Q-Plot of Residuals")
pylab.show()
#######################


######################
# Degree one polyfit #
######################
fit1 = np.polyfit(x=x0, y=y, deg=1)
yHat1 = yHatFunc(x=x0, betas=fit1)
res1 = y - yHat1

    # Plot fit
yHatLine1 = lineFunc(x=x0, betas=fit1)
plt.plot(yHatLine1[0], yHatLine1[1], color='r')
plt.scatter(x=x0, y=y)
plt.title("Fit 1: y and yHat")
plt.show()

    # Plot residuals
zeroLine1 = lineFunc(x=x0, betas=np.zeros(1))
plt.plot(zeroLine1[0], zeroLine1[1], color='r')
plt.scatter(x=x0, y=res1)
plt.title("Fit 1: residuals per x")
plt.show()

    # Histogram Residuals
plt.hist(res1)
plt.title("Fit 1: Residual Frequencies")
plt.show()

    # QQ Plot Residuals
zRes1 = (res1 - np.mean(res1)) / np.std(res1)
sm.qqplot(zRes1, line='45')
pylab.title("Fit 1: Q-Q-Plot of Residuals")
pylab.show()
######################


######################
# Degree two polyfit #
######################
fit2 = np.polyfit(x=x0, y=y, deg=2)
yHat2 = yHatFunc(x=x0, betas=fit2)
res2 = y - yHat2

    # Plot fit
yHatLine2 = lineFunc(x=x0, betas=fit2)
plt.plot(yHatLine2[0], yHatLine2[1], color='r')
plt.scatter(x=x0, y=y)
plt.title("Fit 2: y and yHat")
plt.show()

    # Plot residuals
zeroLine2 = lineFunc(x=x0, betas=np.zeros(1))
plt.plot(zeroLine2[0], zeroLine2[1], color='r')
plt.scatter(x=x0, y=res2)
plt.title("Fit 2: residuals per x")
plt.show()

    # Histogram Residuals
plt.hist(res2)
plt.title("Fit 2: Residual Frequencies")
plt.show()

    # QQ Plot Residuals
zRes2 = (res2 - np.mean(res2)) / np.std(res2)
sm.qqplot(zRes2, line='45')
pylab.title("Fit 2: Q-Q-Plot of Residuals")
pylab.show()
######################


print("hi")
