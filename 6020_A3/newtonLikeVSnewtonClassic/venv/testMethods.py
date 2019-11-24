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
import scipy.stats as stats
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Newton import NewtonRegFunc
from Newton_Like import NewtonLikeFunc
from BisectionMethod import findXzero, findFzero
import globals

globals.initGlobal()


def f(x, a, b, delta):
    fx = a ** 2 / ((b ** 2) + x) ** 2
    fx = np.sum(fx) - delta
    return fx


arrayNsizes = [10, 100, 1000, 10000]
arrayNsizes = np.asarray(arrayNsizes)
deltaSizes = [1e-1, 1e-2, 1e-4, 1e-8]
deltaSizes = np.asarray(deltaSizes)


jLim = np.size(arrayNsizes)
kLim = 1000
randMax = 1000
loopCount = 0
iterationSplit = int(kLim * jLim / 2)
arrayIterations_0 = np.zeros((iterationSplit, 2))
arrayIterations_x0 = copy.deepcopy(arrayIterations_0)


# outer test loop to test four times at each array size
i = 0
while i < 1:

    # inner test loop to test various array sizes
    j = 0
    while j < jLim:

        k = 0
        while k < kLim:
            arrayN = arrayNsizes[j]
            # arrayN = 10
            a = np.random.uniform(0, randMax, arrayN)
            b = a

            delta = deltaSizes[i]
            f_tol = 1e-16
            iterations = 0
            iterationLim = 10000

            if k < 2:
                findXzeroArr = findXzero(f, a, b, delta)
                x0 = findXzeroArr[0]
                iterationStart = findXzeroArr[1]
            else:
                x0 = 0
                iterationStart = 0

            # x0 = findXzero(f, a, b, delta)

            # print("Newton Regular:")
            NewtonRegIterations = NewtonRegFunc(x0, a, b, delta, f_tol, iterationLim, iterationStart=iterationStart, plotColor=0)

            # print("Newton Like:")
            NewtonLikeIterations = NewtonLikeFunc(x0, a, b, delta, f_tol, iterationLim, iterationStart=iterationStart)

            if j < 2:
                arrayIterations_x0[loopCount, 0] = NewtonRegIterations
                arrayIterations_x0[loopCount, 1] = NewtonLikeIterations

            else:
                temp = loopCount - iterationSplit
                arrayIterations_0[temp, 0] = NewtonRegIterations
                arrayIterations_0[temp, 1] = NewtonLikeIterations


            # print("Bisection Method:")
            # findFzero(f, a, b, delta, f_tol)

            # plt.title("Iterations Per Method. Delta: %.1E . N: %d" % (delta, arrayN))
            # plt.xlabel("Iterations")
            # plt.ylabel("Log Residuals")

            # fileName = "figures/Delta%1.E N%d.png" % (delta, arrayN)
            # plt.savefig(fileName, dpi=72, bbox_inches='tight')

            # plt.show()

            k += 1
            loopCount += 1

        j += 1

    i += 1

mu_iterations_x0_NReg = np.mean(arrayIterations_x0[:, 0])
sd_iterations_x0_NReg = np.std(arrayIterations_x0[:, 0])

mu_iterations_x0_NLike = np.mean(arrayIterations_x0[:, 1])
sd_iterations_x0_NLike = np.std(arrayIterations_x0[:, 1])

mu_iterations_0_NReg = np.mean(arrayIterations_0[:, 0])
sd_iterations_0_NReg = np.std(arrayIterations_0[:, 0])

mu_iterations_0_NLike = np.mean(arrayIterations_0[:, 1])
sd_iterations_0_NLike = np.std(arrayIterations_0[:, 1])

plt.hist(arrayIterations_x0[:, 0])
plt.title("Newton-Classic with x0 = findXzero().")
plt.xlabel("iterations")
plt.ylabel("frequency")
plt.xlim(0, )
plt.ylim(0, iterationSplit)
plt.show()

plt.hist(arrayIterations_x0[:, 1])
plt.title("Newton-Like with x0 = findXzero().")
plt.xlabel("iterations")
plt.ylabel("frequency")
plt.ylim(0, iterationSplit)
plt.xlim(0, )
plt.show()

plt.hist(arrayIterations_0[:, 0])
plt.xlabel("iterations")
plt.ylabel("frequency")
plt.title("Newton-Classic with x0 = 0.")
plt.ylim(0, iterationSplit)
plt.xlim(0, )
plt.show()

plt.hist(arrayIterations_0[:, 1])
plt.xlabel("iterations")
plt.ylabel("frequency")
plt.title("Newton-Like with x0 = 0.")
plt.ylim(0, iterationSplit)
plt.xlim(0, )
plt.show()

plt.show()

print("")

# tempArr = stats.poisson(arrayIterations_x0[:, 0])
# mu_iterations_x0_NReg = tempArr[0]
# var_iterations_x0_NReg = tempArr[1]
#
# tempArr = stats.poisson(arrayIterations_x0[:, 1])
# mu_iterations_x0_NLike = tempArr[0]
# var_iterations_x0_NLike = tempArr[1]
#
# tempArr = stats.poisson(arrayIterations_0[:, 1])
# mu_iterations_0_NReg = tempArr[0]
# var_iterations_0_NReg = tempArr[1]
#
# tempArr = stats.poisson(arrayIterations_0[:, 1])
# mu_iterations_0_NLike = tempArr[0]
# var_iterations_0_NLike = tempArr[1]

print("Total Tests Performed: %d" % iterationSplit)

print("NReg (x0) mu = %f , sd = % f" % (mu_iterations_x0_NReg, sd_iterations_x0_NReg))
print("NLike (x0) mu = %f , sd = % f" % (mu_iterations_x0_NLike, sd_iterations_x0_NLike))

print("NReg (0) mu = %f , sd = % f" % (mu_iterations_0_NReg, sd_iterations_0_NReg))
print("NLike (0) mu = %f , sd = % f" % (mu_iterations_0_NLike, sd_iterations_0_NLike))

print("hi")
