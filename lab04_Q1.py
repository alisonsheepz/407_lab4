# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:53:35 2018

@author: xxu02
"""
#import SolveLinear as sl
import numpy as np
from numpy import copy

A = np.array([[0,0,4,1],[3,0,-1,-1],[1,-4,0,5],[2,-2,1,3]])
V = [[2],[-1],[-2],[1]]
N = 4

from numpy.linalg import solve
#print(solve(A,V))
#print(sl.PartialPivot(A,V))
print(A,'\n')

for m in range(N):
    if A[m, m] == 0:
        
        val_buff = 0
        index_buff = 0
        for n in range(m + 1, N):
            if np.abs(A[n, m]) > np.abs(val_buff):
                val_buff = A[n, m]
                index_buff = n
        
        print('m is', m, 'n is', index_buff)
        A[index_buff, :], A[m, :] = copy(A[m, :]), copy(A[index_buff, :])
        print(A)
