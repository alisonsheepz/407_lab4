# SolveLinear.py
# Python module for PHY407
# Paul Kushner, 2015-09-26
# Modifications by Nicolas Grisouard, 2018-09-26
# This module contains useful routines for solving linear systems of equations.
# Based on gausselim.py from Newman
from numpy import empty, copy
import numpy as np
# The following will be useful for partial pivoting
# from numpy import empty, copy

#


def GaussElim(A_in, v_in):
    """Implement Gaussian Elimination. This should be non-destructive for input
    arrays, so we will copy A and v to
    temporary variables
    IN:
    A_in, the matrix to pivot and triangularize
    v_in, the RHS vector
    OUT:
    x, the vector solution of A_in x = v_in """
    # copy A and v to temporary variables using copy command
    A = copy(A_in)
    v = copy(v_in)
    N = len(v)

    for m in range(N):
        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult*A[m, :]
            v[i] -= mult*v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = empty(N, dtype=v.dtype)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i]*x[i]
    return x


def PartialPivot(A_in, v_in):
    """ In this function, code the partial pivot (see Newman p. 222) """
    
    # copy A and v to temporary variables using copy command
    A = copy(A_in)
    v = copy(v_in)
    N = len(v)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Q1a
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Pivotting
    for m in range(N):
        if A[m, m] == 0.:
            
            # The buffers to record the index and the max value
            val_buff = 0.
            index_buff = 0
            
            for n in range(m + 1, N):
                
                # Find the one farthest from 0 
                if np.abs(A[n, m]) > np.abs(val_buff):
                    val_buff = A[n, m]
                    index_buff = n
            
            # Perform the row interchanging
            A[index_buff, :], A[m, :] = copy(A[m, :]), copy(A[index_buff, :])
            v[index_buff, :], v[m, :] = copy(v[m, :]), copy(v[index_buff, :])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    End of Q1a
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    for m in range(N):
        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult*A[m, :]
            v[i] -= mult*v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = empty(N, dtype=v.dtype)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i]*x[i]
    return x
        
    print('hello')
