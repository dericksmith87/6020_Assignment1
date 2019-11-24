import math
import scipy
import scipy.linalg
import copy

def f(x):  # First test function: two equations
    y = scipy.zeros((2, 1))  # Initialize output vector
    y[0] = 2.0 * math.exp(x[0] * x[1]) - 2.0 * x[0] + 2.0 * x[1] - 2  # Assign function values
    y[1] = x[0] ** 5 + x[0] * x[1] ** 5 - 2.0 * x[1]
    return y


def Df(x):  # Jacobian for first test case
    J = scipy.zeros((2, 2))  # Initialize as 2-by-2 array
    J[0, 0] = 2.0 * x[1] * math.exp(x[0] * x[1]) - 2.0  # Assign values
    J[0, 1] = 2.0 * x[0] * math.exp(x[0] * x[1]) + 2.0
    J[1, 0] = 5.0 * x[0] ** 4 + x[1] ** 5
    J[1, 1] = 5.0 * x[0] * x[1] ** 4 - 2.0
    return J


def fN(x, N, l):  # Second test function with parameter l and n equations
    # y = scipy.zeros((N, 1))  # Initialize output vector
    temp = scipy.zeros((N, 1))  # Initialize output vector
    y = copy.deepcopy(temp)
    # for i in range(0, N - 1):  # Assign function values in loop
    try:
        for i in range(0, N):  # Assign function values in loop
            y[i] = x[i] - math.exp(l * math.cos((i + 1.0) * scipy.sum(x)))
        return y
    except ValueError:
        print("Value Error")
        return temp #?



def DfN(x, N, l):  # Jacobian for second test case
    temp = scipy.identity(N)  # Initialize as N-by-N array
    J = copy.deepcopy(temp)  # Initialize as N-by-N array
    S = sum(x)  # Compute sum
    try:
        for i in range(0, N):  # Assign values
            J[i, :] = J[i, :] + (i + 1.0) * l * math.sin((i + 1.0) * S) * math.exp(l * math.cos((i + 1.0) * S)) * scipy.ones((1, N))
        return J
    except TypeError:
        print("Type Error")
        return temp
