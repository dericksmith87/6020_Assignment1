# Newton iteration. Programmed in class in Lecture 12. 
import numpy as np
from copy import copy

def Newton(f,fp,x0,eps_f,eps_x,N):
    x = copy(x0)
    for i in range(N):
        r = f(x)
        dx = -r/fp(x)
        x += dx

        print('%d %e %e %e' % (i,r,dx,x))
        if abs(r) < eps_f and abs(dx) < eps_x:
            return x
    print('No convergence after %d iterations!' % (N))
    
