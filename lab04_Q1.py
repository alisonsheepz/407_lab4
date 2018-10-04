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
#number_of_tests = 30
#
#LU_time, PP_time, GE_time = np.zeros(number_of_tests),\
# np.zeros(number_of_tests), np.zeros(number_of_tests)
# 
#LU_err, PP_err, GE_err = np.zeros(number_of_tests),\
# np.zeros(number_of_tests), np.zeros(number_of_tests)
# 
#Test_points = range(number_of_tests)
#
#for n in Test_points:
#    
#    # Since n starts at 0 and we dont want 0 by 0 arrays
#    N = (n + 1) * 10
#    V = rand(N)
#    A = rand(N, N)
#    
#    v_sol = np.dot(A, V)
#    
#    # repeat 10 timesand time it
#    start = time()
#    for dump in range(10):
#       result_buffer = LU_decomp(A,V)
#       
#    end = time()
#    LU_time[n] = end - start
#    LU_err[n] = np.mean(np.abs(result_buffer - v_sol))
#    
#
#    # repeat 10 timesand time it
#    start = time()
#    for dump in range(10):
#       result_buffer = sl.PartialPivot(A,V)
#       
#    end = time()
#    PP_time[n] = end - start
#    PP_err[n] = np.mean(np.abs(result_buffer - v_sol))
#
#
#    # repeat 10 timesand time it
#    start = time()
#    for dump in range(10):
#       result_buffer = sl.GaussElim(A,V)
#       
#    end = time()
#    GE_time[n] = end - start
#    GE_err[n] = np.mean(np.abs(result_buffer - v_sol))
#    
#    
## Plotting out the timings on the log  scale
#plt.figure()
#plt.yscale('log')
#plt.plot(Test_points, LU_time, linewidth = 0.8, label = 'LU time')
#plt.plot(Test_points, GE_time, linewidth = 0.8, label = 'GE time')
#plt.plot(Test_points, PP_time, linewidth = 0.8, label = 'PP time')
#plt.title('Computing time of different methods vs # of sample points')
#plt.xlabel('# of data samples points')
#plt.ylabel('time(s)')
#plt.legend()
#plt.savefig('time.pdf')
#plt.show()
#
## Plotting out the errors on the log log scale
#plt.figure()
#plt.yscale('log')
#plt.plot(Test_points, LU_err, linewidth = 0.8, label = 'LU error')
#plt.plot(Test_points, GE_err, linewidth = 0.8, label = 'GE error')
#plt.plot(Test_points, PP_err, linewidth = 0.8, label = 'PP error')
#plt.title('Error of different methods vs # of sample points')
#plt.xlabel('# of data samples points')
#plt.legend()
#plt.savefig('error.pdf')
#plt.show()
#
#print('As we can tell, for the error graphy, all three lines overlaps,\
#which meaning they are the same. Therefore, there is no difference in \
#error \n')
#
#print('We can also tell that the LU_decomposition is significantly faster than\
#the partial pivot method or the GaussElimination method,\
#even though the computing time increases when the size of the matrix \
#increases. Also, it is worth noting that the partial pivotting method\
#is expected to be slower than the GaussElimination method, as it does\
#extra array manipulation. However; the y axis is logarithmic, we can\'t\
#tell that from the graph.\n')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Q1b
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# constants from the text
R1 = R3 = R5 = 1000. # ohm
R2 = R4 = R6 = 2000.
C1 = 1e-6
C2 = 0.5 * 1e-6
x_plus = 3.
w = 1000.
L = 2.
V = np.array([1./R1, 1./R2, 1./R3], dtype=complex)
A = np.array(\
     [[1./R1 + 1./R4 + w*C1*1j, -w*C1                       , 0   ],
      [-w*C1                 ,1./R2 + 1./R5 + 1j*w*(C1 + C2),-w*C2],
      [0                     ,-w*C2                        ,1./R3 + 1./R6 + 1j*w*C2 ]]\
      ,dtype=complex)
A_l = np.array(\
     [[1./R1 + 1./R4 + w*C1*1j, -w*C1                       , 0   ],
      [-w*C1                 ,1./R2 + 1./R5 + 1j*w*(C1 + C2),-w*C2],
      [0                     ,-w*C2                        ,1./R3 + 1./L + 1j*w*C2 ]]\
      ,dtype=complex)

# it has x1 x2 x3
X = sl.PartialPivot(A,V)

print('Amplitude of X1 is ', np.abs(X[0]), 'v')
print('Amplitude of X2 is ', np.abs(X[1]), 'v')
print('Amplitude of X3 is ', np.abs(X[2]), 'v')
print('\n')
print('Phase of X1 is ', np.angle(X[0], deg = True),'degrees')
print('Phase of X2 is ', np.angle(X[1], deg = True),'degrees')
print('Phase of X3 is ', np.angle(X[2], deg = True),'degrees')

# have 2 priods
T = np.linspace(0, np.pi * 2 / 500, 1000)

# since we only need the real part, we can get rid of the imaginary part of the 
# result of eulers equation
V1_out = np.abs(X[0]) * np.cos(w*T) - np.angle(X[0]) * np.sin(w*T)
V2_out = np.abs(X[1]) * np.cos(w*T) - np.angle(X[1]) * np.sin(w*T)
V3_out = np.abs(X[2]) * np.cos(w*T) - np.angle(X[2]) * np.sin(w*T)

# plottinh
plt.figure()
plt.plot(T, V1_out, label = 'V1')
plt.plot(T, V2_out, label = 'V2')
plt.plot(T, V3_out, label = 'V3')
plt.title('V vs t')
plt.xlabel('Time in seconds')
plt.ylabel('Voltage in volt')
plt.legend()
plt.show()

# it has x1 x2 x3
X = sl.PartialPivot(A_l,V)

print('After replacing R6 with L')
print('Amplitude of X1 is ', np.abs(X[0]), 'v')
print('Amplitude of X2 is ', np.abs(X[1]), 'v')
print('Amplitude of X3 is ', np.abs(X[2]), 'v')
print('\n')
print('Phase of X1 is ', np.angle(X[0], deg = True),'degrees')
print('Phase of X2 is ', np.angle(X[1], deg = True),'degrees')
print('Phase of X3 is ', np.angle(X[2], deg = True),'degrees')

# since we only need the real part, we can get rid of the imaginary part of the 
# result of eulers equation
V1_out = np.abs(X[0]) * np.cos(w*T) - np.angle(X[0]) * np.sin(w*T)
V2_out = np.abs(X[1]) * np.cos(w*T) - np.angle(X[1]) * np.sin(w*T)
V3_out = np.abs(X[2]) * np.cos(w*T) - np.angle(X[2]) * np.sin(w*T)


# plottinh
plt.figure()
plt.plot(T, V1_out, label = 'V1')
plt.plot(T, V2_out, label = 'V2')
plt.plot(T, V3_out, label = 'V3')
plt.title('V vs t with inductor')
plt.xlabel('Time in seconds')
plt.ylabel('Voltage in volt')
plt.legend()
plt.show()