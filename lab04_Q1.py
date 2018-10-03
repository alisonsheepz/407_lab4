# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:53:35 2018

@author: xxu02
"""
import SolveLinear as sl
import numpy as np
from numpy.linalg import solve as LU_decomp
from time import time
import matplotlib.pyplot as plt

A = np.array([[0,0,4,1],[3,0,-1,-1],[1,-4,0,5],[2,-2,1,3]])
V = [[2],[-1],[-2],[1]]
N = 4

print(np.random.randn(2,2))

print(LU_decomp(A,V))
#print(sl.PartialPivot(A,V))
print(sl.GaussElim(A,V))
