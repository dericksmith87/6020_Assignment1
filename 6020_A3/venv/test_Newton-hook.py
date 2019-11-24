# Test script for Newton-hook iteration. Programmed in lecture 12.
import numpy as np
from Newton_hook import Newton_hook
from functions import fN, DfN
# For comparison:
from Newton_Raphson import Newton_Raphson
# For plotting:
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Convergece criteria:
tol_f = 1e-12
tol_x = 1e-12

# Initial and minimal size of trust region:
delta = 0.001
delta_min = 1e-10

# Factors to decrease/increase the trust region:
alpha = 0.8
beta = 1.2

# Number of equations to consider and system parameter:
# N = 25
# lam = 5e-1

# Initial guess (exact solution for lam = 0)
# x0 = np.ones((N, 1))

# Maximal number of iterations:
N_max = 1000


# Auxiliary definition of the test problem with only one argument:
def G(x):
    return fN(x, N, lam)


def DG(x):
    return DfN(x, N, lam)





numEqsArr = [2, 4, 8, 16, 32]
lambdaValArr = [5e-1, 5e-2, 5e-3, 5e-4, 5e-5]



loop1_max = np.size(numEqsArr)
loop2_max = np.size(lambdaValArr)
loop3_max = 5


i = 0
while i < loop1_max:

    numEqs = numEqsArr[i]
    N = numEqs
    j = 0
    while j < loop2_max:

        lambdaVal = lambdaValArr[j]
        lam = lambdaVal
        k = 0
        while k < loop3_max:

            if k > 0:
                x0 = x0 + np.random.normal(loc=0, scale=delta_min, size=numEqs)
            else:
                x0 = np.ones((numEqs, 1))

            x1, diag1 = Newton_hook(x0, lambda x: fN(x, N, lam), lambda x: DfN(x, N, lam), delta, N_max, tol_f, tol_x, alpha, beta, delta_min)
            # x1, diag1 = Newton_hook(x0, G, DG, delta, N_max, tol_f, tol_x, alpha, beta, delta_min, finiteDiff=True)
            x1, diag1 = Newton_hook(x0, lambda x: fN(x, N, lam), lambda x: DfN(x, N, lam), delta, N_max, tol_f, tol_x, alpha, beta, delta_min)
            # print(x1)

            x2, diag2 = Newton_Raphson(lambda x: fN(x, N, lam), lambda x: DfN(x, N, lam), x0, tol_f, tol_x, N)
            # x2, diag2 = Newton_Raphson(G, DG, x0, tol_f, tol_x, N)
            # print(x2)



            # diag.append([k, res, size, delta])
            # [,0] = number of iterations
            # [,1] = residual
            # [,2] = dx (size of step)
            # [,3] = delta (size of trust region)
            try:
                plotColor = ["red", "blue", "green"]
                plt.loglog(diag1[:, 0], diag1[:, 1], color=plotColor[0])
                plt.loglog(diag1[:, 0], diag1[:, 2], color=plotColor[1])
                plt.loglog(diag1[:, 0], diag1[:, 3], color=plotColor[2])
                plt.title("Newton-Hook Settings: N = %d ; lambda = %.1E; iteration = %d" % (numEqs, lambdaVal, k))
                plt.show()
            except (UserWarning, ValueError):
                print("Negative values cannot be log-scaled")

            try:
                # diag.append([i, res, err])
                # [,0] = number of iterations?
                # [,1] = residual
                # [,2] = error (size of step)
                plotColor = cm.rainbow(np.linspace(0, 1, 3))
                plt.loglog(diag2[:, 0], diag2[:, 1], color=plotColor[0])
                plt.loglog(diag2[:, 0], diag2[:, 2], color=plotColor[1])
                plt.title("Newton-Raphson Settings: N = %d ; lambda = %.1E; iteration = %d" % (numEqs, lambdaVal, k))
                plt.show()
            except (UserWarning, ValueError):
                print("Negative values cannot be log-scaled")
            # plt.title("Iterations Per Method. Delta: %.1E . N: %d" % (delta, arrayN))
            # plt.xlabel("Iterations")
            # plt.ylabel("Log Residuals")

            # fileName = "figures/Delta%1.E N%d.png" % (delta, arrayN)
            # plt.savefig(fileName, dpi=72, bbox_inches='tight')

            # plt.show()














            k += 1

        j += 1

    i += 1



