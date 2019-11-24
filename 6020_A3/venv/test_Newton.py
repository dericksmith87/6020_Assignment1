# Test script for Newton iteration. Programmed in lecture 12.

import numpy as np
from Newton import Newton

def f(x):
    return np.sin(x*x)

def fp(x):
    return 2*x*np.cos(x*x)

eps_f = 1e-12
eps_x = 1e-12
N = 5

x0 = 4.5

x = Newton(f,fp,x0,eps_f,eps_x,N)
