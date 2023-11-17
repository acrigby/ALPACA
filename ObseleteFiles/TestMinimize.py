# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:54:47 2023

@author: aidan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import minimize, basinhopping

x = np.arange(0,100,0.01)
    
consts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5]
FF = []
n = 0

# for point in x:
#     FFp = 0
#     if point < 20 :
#         FF.append(0)
#     elif point > 20 + P: 
#         FF.append(consts[(len(consts)-1)])
#     else:
#         for i in range(1,int((len(consts)-2)/2)+1,1):
#             FFp += consts[i-1]*np.sin((2*np.pi*(point-20)*i)/P) + consts[i+int((len(consts)-2)/2)]*np.cos((2*np.pi*(point-20)*i)/P)
#         FFp += consts[(len(consts)-1)]
#         FF.append(FFp)

# plt.plot(x,FF, label = 'Guess Combination')
# plt.legend()

def funct(consts):
    global n
    P = 20
    x = np.arange(0,100,0.01)
    FF = []

    for point in x:
        FFp = 0
        if point < 20 :
            FF.append(0)
        elif point > 20 + P: 
            FF.append(-0.5)
        else:
            for i in range(1,int((len(consts)-1)/2)+1,1):
                FFp += consts[i-1]*np.sin((2*np.pi*(point-20)*i)/P) + consts[i+int((len(consts)-1)/2)-1]*np.cos((2*np.pi*(point-20)*i)/P)
            FFp += consts[(len(consts)-1)]
            FF.append(FFp)

    result = 0
    for i in range(len(FF)):
        if 0 <= x[i]< 20:
            result += abs(FF[i])
        elif 20 <= x[i]< 25:
            result += abs(FF[i]-((0.14*x[i])-2.8))
        elif 25 <= x[i]< 35:
            result += abs(FF[i]-((-0.12*x[i])+3.7))
        else:
            result += abs(FF[i]+0.5)
    n += 1
    print(result)
    return result

res = basinhopping(funct, consts, niter=1,T=1000,minimizer_kwargs={"method":"Powell", "tol":1e-2})
print(n)
print(res.x)
consts = res.x
FF = []   

score = funct(consts) 
print(score)

for point in x:
    P = 20
    FFp = 0
    if point < 20 :
        FF.append(0)
    elif point > 20 + P: 
        FF.append(-0.5)
    else:
        for i in range(1,int((len(consts)-1)/2)+1,1):
            FFp += consts[i-1]*np.sin((2*np.pi*(point-20)*i)/P) + consts[i+int((len(consts)-1)/2)-1]*np.cos((2*np.pi*(point-20)*i)/P)
        FFp += consts[(len(consts)-1)]
        FF.append(FFp)

y = []
plt.plot(x,FF, label = 'Optimised Combination')
for i in range(len(x)):
    if 0 <= x[i]< 20:
        y.append(0)
    elif 20 <= x[i]< 25:
        y.append((0.14*x[i])-2.8)
    elif 25 <= x[i]< 35:
        y.append((-0.12*x[i])+3.7)
    else:
        y.append(-0.5)
        
plt.plot(x,y, label = 'Desired function')
plt.legend()
