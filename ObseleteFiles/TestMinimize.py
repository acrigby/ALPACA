# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:54:47 2023

@author: aidan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import minimize

x = np.arange(0,100,0.01)
    
consts = [1,1,1,1]
FF = []

for point in x:
    FFp = 0
    for i in range(0,len(consts)-1,1):
        FFp += consts[i]*jv(i,(point/consts[len(consts)-1]))
    FF.append(FFp)

plt.plot(x,FF, label = 'Guess Combination')
plt.legend()

def funct(consts):
    x = np.arange(0,100,0.01)
    FF = []

    for point in x:
        FFp = 0
        for i in range(0,len(consts)-1,1):
            FFp += consts[i]*jv(i,(point/consts[len(consts)-1]))
        FF.append(FFp)

    result = 0
    for i in range(len(FF)):
        result += abs(FF[i]-np.sin(x[i]))
    
    return result


res = minimize(funct, consts, method='Nelder-Mead', tol=1e-3)
print(res.x)
consts = res.x
FF = []   

score = funct(consts) 
print(score)

for point in x:
    FFp = 0
    for i in range(0,len(consts)-1,1):
        FFp += consts[i]*jv(i,(point/consts[len(consts)-1]))
    FF.append(FFp)

plt.plot(x,FF, label = 'Optimised Combination')
y = np.sin(x)
plt.plot(x,y, label = 'Sin function')
plt.legend()
