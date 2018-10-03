# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:53:35 2018

@author: xxu02
"""
A = [[3,5,6,2],[2,0,1,3],[4,1,5,6],[-5,3,5,3]]
V = [[24],[7],[13],[-7]]

from numpy.linalg import solve

print(solve(A,V))