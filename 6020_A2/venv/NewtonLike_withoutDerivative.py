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


def f(x, a, b, delta):
    fx = a ** 2 / ((b ** 2) + x) ** 2
    fx = np.sum(fx) - delta
    return fx


def fp(x, a, b):
    fpx = -2 * a ** 2 / ((b ** 2) + x) ** 3
    fpx = np.sum(fpx)
    return fpx


# def pfp(x, a, b, step, f):  # pseudo fp
#     pfp = f((x + step), a, b) - f(x, a, b)
#     pfp = pfp / step
#     pfp = np.sum(pfp)
#     return pfp


def g(xkr, a, b, delta):
# def g(xkr, f, fp):

    fk = f(xkr, a, b, delta)
    fpk = fp(xkr, a, b)

    xk1 = -fk / fpk + xkr
    fk1 = f(xk1, a, b, delta)

    B = (fk1 * xk1 - fk * xkr) / (fk - fk1)
    A = fk * (B + xkr)

    # xkr = A / (fk1 / 2 + delta) - B
    xkr = A / (fk1 + delta) - B

    return xkr


x0 = 0
# step = 1e-12
# arrayN = 3
a = [-0.62599985, 1.48611463, -6.25436486]
a = np.asarray(a)
# a = np.random.uniform(-10, 10, arrayN)
b = [-6.12939018, -8.30181068, -1.56093715]
b = np.asarray(b)
# b = np.random.uniform(-10, 10, arrayN)

delta = 1e-5
f0 = f(0, a, b, delta)
# delta = np.random.uniform(1e-14, np.min([f0, 1/arrayN]), 1)
f_tol = 1e-16
iterations = 0
iterationLim = 100

# f1 = 1
xkr = x0

while iterations < iterationLim:

    f1 = f(xkr, a, b, delta)
    print("current f(x) = %f" % f1)
    if f1 < 0:
        print("Error")
        break
    # print(iterations)
    if np.abs(f1) <= f_tol:
        break

    fp1 = fp(xkr, a, b)
    # fp1 = pfp(xk, a, b, step, f)

    xkr = g(xkr, a, b, delta)
    print("current x = %f" % xkr)

    iterations += 1

print("hi")


