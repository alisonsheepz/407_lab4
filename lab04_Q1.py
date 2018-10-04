# -*- coding: utf-8 -*-
"""
Lab 04 Q1
"""
import SolveLinear as sl
import numpy as np
from numpy.linalg import solve as LU_decomp
from numpy.random import rand
from time import time
import matplotlib.pyplot as plt
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Q1b
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
number_of_tests = 30

LU_time, PP_time, GE_time = np.zeros(number_of_tests),\
 np.zeros(number_of_tests), np.zeros(number_of_tests)
 
LU_err, PP_err, GE_err = np.zeros(number_of_tests),\
 np.zeros(number_of_tests), np.zeros(number_of_tests)
#LU_results, PP_results, GE_results, cor_results = np.zeros((number_of_tests, 4)),\
# np.zeros((number_of_tests, 4)), np.zeros((number_of_tests, 4)), np.zeros((number_of_tests, 4))

Test_points = range(number_of_tests)

for n in Test_points:
    
    # Since n starts at 0 and we dont want 0 by 0 arrays
    N = (n + 1) * 10
    V = rand(N)
    A = rand(N, N)
    
    v_sol = np.dot(A, V)
    
    # repeat 10 timesand time it
    start = time()
    for dump in range(10):
       result_buffer = LU_decomp(A,V)
       
    end = time()
    LU_time[n] = end - start
    LU_err[n] = np.mean(np.abs(result_buffer - v_sol))
    

    # repeat 10 timesand time it
    start = time()
    for dump in range(10):
       result_buffer = sl.PartialPivot(A,V)
       
    end = time()
    PP_time[n] = end - start
    PP_err[n] = np.mean(np.abs(result_buffer - v_sol))


    # repeat 10 timesand time it
    start = time()
    for dump in range(10):
       result_buffer = sl.GaussElim(A,V)
       
    end = time()
    GE_time[n] = end - start
    GE_err[n] = np.mean(np.abs(result_buffer - v_sol))
    
    
# Plotting out the timings on the log  scale
plt.figure()
plt.yscale('log')
plt.plot(Test_points, LU_time, linewidth = 0.8, label = 'LU time')
plt.plot(Test_points, GE_time, linewidth = 0.8, label = 'GE time')
plt.plot(Test_points, PP_time, linewidth = 0.8, label = 'PP time')
plt.title('Computing time of different methods vs # of sample points')
plt.xlabel('# of data samples points')
plt.ylabel('time(s)')
plt.legend()
plt.savefig('time.pdf')
plt.show()

# Plotting out the errors on the log log scale
plt.figure()
plt.yscale('log')
plt.plot(Test_points, LU_err, linewidth = 0.8, label = 'LU error')
plt.plot(Test_points, GE_err, linewidth = 0.8, label = 'GE error')
plt.plot(Test_points, PP_err, linewidth = 0.8, label = 'PP error')
plt.title('Error of different methods vs # of sample points')
plt.xlabel('# of data samples points')
plt.legend()
plt.savefig('error.pdf')
plt.show()

print('As we can tell, for the error graphy, all three lines overlaps,\
which meaning they are the same. Therefore, there is no difference in \
error \n')

print('We can also tell that the LU_decomposition is significantly faster than\
the partial pivot method or the GaussElimination method,\
even though the computing time increases when the size of the matrix \
increases. Also, it is worth noting that the partial pivotting method\
is expected to be slower than the GaussElimination method, as it does\
extra array manipulation. However; the y axis is logarithmic, we can\'t\
tell that from the graph.\n')
