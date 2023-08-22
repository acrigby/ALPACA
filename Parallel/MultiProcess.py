# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:28:04 2023

@author: localuser
"""

from multiprocessing import Pool
import subprocess
from DyMat.DyMat import DyMatFile
import matplotlib.pyplot as plt

def f(inputfile):
    print(inputfile)
    subprocess.check_call("C:/Users/localuser/Documents/GitHub/ALPACA/Parallel/dymosim " + inputfile )
    r = DyMatFile("dsres.mat")
    variables = ["BOP.sensor_pT.T","BOP.sensor_pT.p","BOP.sensor_T2.T","BOP.TCV.dp","pump_SimpleMassFlow1.m_flow", "ramp.y", "boundary.Q_flow_ext","BOP.sensorW.W","FeedForward.y"]
    t = r.abscissa(variables[0])[0]
    Pout = r.data("BOP.sensor_pT.p")
    
    return(t, Pout)


if __name__ == '__main__':
    with Pool(3) as p:
        Pouts1, Pouts2 = p.map(f, ['dsin.txt', 'dsin_Copy.txt'])
        plt.plot(Pouts1[0], Pouts1[1])
        plt.plot(Pouts2[0], Pouts2[1])
    
    plt.show()