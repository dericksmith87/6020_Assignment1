# Test system on N nonlinear equations. By L. van Veen, OnTechU, 2019.
import numpy as np
def fN(x,N,l):                                               # Test function with parameter l and N equations
    y = np.zeros((N,1))                                      # Initialize output vector
    for i in range(0,N-1):                                   # Assign function values in loop
        y[i] = x[i] - np.exp(l * np.cos((i+1.0)*np.sum(x)))
    return y

def DfN(x,N,l):                                              # Jacobian for second test case
    J = np.identity(N)                                       # Initialize as N-by-N array
    S = np.sum(x)                                            # Compute sum
    for i in range(0,N):                                     # Assign values
        J[i,:] = J[i,:] + (i+1.0) * l * np.sin((i+1.0)*S) * np.exp(l * np.cos((i+1.0)*S)) * np.ones((1,N))
    return J
