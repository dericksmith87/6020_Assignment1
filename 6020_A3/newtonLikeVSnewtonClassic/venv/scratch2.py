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
import matplotlib.cm as cm
import globals
from globals import objFunc

globals.initGlobal()


def f(x):
    result = x ** 2
    return result


h = 1e-10

df = objFunc(f, h)

x = 1

result = df(x)

print("hi")

