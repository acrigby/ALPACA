# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:25:57 2023

@author: localuser
"""

import concurrent.futures
import subprocess
from DyMat.DyMat import DyMatFile
import matplotlib.pyplot as plt
import matplotlib
import itertools
import concurrent.futures
import pathlib
import re
import shutil
import os
import numpy as np
import time

st = time.time()

print(st)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

files = ['dsfinal','dsfinal','dsfinal','dsfinal','dsfinal','dsfinal','dsfinal','dsfinal']


def worker(inputfile, src, target, taskid):

    os.makedirs(target, exist_ok=True)
    
    path1 = pathlib.Path(str(src) + "/" + inputfile + ".txt")
    path2 = pathlib.Path(target)
    path3 = pathlib.Path(str(src)+ "/" +'dymosim.exe')
    path4 = pathlib.Path(str(target)+ "/" +'dymosim.exe')

    shutil.copy(path1, path2)
    shutil.copy(path3, path2)

    #print(path4)
    os.chdir(str(path2))
    os.system('echo %cd%')
    
    temps = []
    feeds = []
    t = []
    t1 = []
    
    for i in range(0,1000,5):
        with open('dsfinal.txt', 'r') as file:
            # read a list of lines into data
            data = file.readlines()
    
        # now change the 2nd line, note that you have to add a newline
        data[8] = '   '+ str(9900+i) + '\n'
        data[9] = '   '+ str(9900+i+5) + '\n'
        data[7781] = '-2       ' + str(np.random.random()*0.5) + '                0                       0 '
        #print(data[7781])
        #print(data[7782])
        # and write everything back
        with open('dsfinal.txt', 'w') as file:
            #print("T Start: " + data[8])
            #print("T End: " + data[9])
            file.writelines(data)
            
            
        subprocess.run(str(path4)+ ' ' + inputfile + ".txt", stdout = subprocess.DEVNULL)
        r = DyMatFile(str(path2)+"/dsres.mat")
        variables = ["BOP.sensor_pT.T","BOP.sensor_pT.p","BOP.sensor_T2.T","BOP.TCV.dp","pump_SimpleMassFlow1.m_flow", "ramp.y", "boundary.Q_flow_ext","BOP.sensorW.W","FeedForward.y"]
        t.extend(r.abscissa(variables[0])[0])
        t1.extend(r.abscissa("FeedForward.y")[0])
        temps.extend(r.data("BOP.sensor_pT.T"))
        feeds.extend(r.data("FeedForward.y"))
        
    fig, axs = plt.subplots(2,1)
    axs[0].plot(t, temps, label = 'Optimised')
    #axs[0].set_xlim(0,max(t))
    axs[0].axhspan(666.15, 680.15, color='red', alpha=0.35)
    axs[0].axhspan(671.15, 675.15, color='green', alpha=0.5)
    #axs.set_xlim(0,200)
    axs[0].legend(ncol=2)
    axs[0].set(ylabel = 'Temperature / K')
    axs[0].set_title('Temperature')
    # axs[0, 1].plot(t1, mdots, 'tab:orange')
    # axs[0, 1].set_title('Pump Mdot')
    axs[1].plot(t1, feeds, 'tab:green')
    #axs[1].set_xlim(min(t),max(t))
    axs[1].set_title('FF Signal')
    # axs[1, 1].plot(t1, TCVdps, 'tab:red')
    # axs[1, 1].set_title('TCV pressure drop')
    fig.tight_layout()
    axs[1].set(xlabel='Time / s')
    #print(feeds)
    fig.savefig('graph.png')
    plt.show() 

def main():
    for i in range(0,7,1):
        listfiles = files[:i]
        # src: The directory where this script is
        src = pathlib.Path(__file__).parent
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for taskid, file in enumerate(listfiles, 1):
                target = pathlib.Path( str(src) + f"/tmp/dir{taskid}")
    
                # Calls `worker` function with parameters path1, path2, ...
                # concurrently
                executor.submit(worker, file, src, target, taskid)
                
        et = time.time()
       
        elapsed_time = et - st
           
        print('Execution time:', elapsed_time, 'seconds')

if __name__ == "__main__":
    main()
    
