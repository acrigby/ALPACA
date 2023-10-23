# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:25:27 2023

@author: aidan
"""

import matplotlib
import matplotlib.pyplot as plt
from dymola.dymola_interface import DymolaInterface
import numpy as np
import os
from scipy.special import jv
from scipy.optimize import minimize
print(os.path.abspath(os.curdir))
os.chdir("..")
print(os.path.abspath(os.curdir))
ALPACApath = os.path.abspath(os.curdir)

f = open("./Utilities/ALPACAASCII.txt", 'r')
print(''.join([line for line in f]))

model = "ControlTests.SteamTurbine_L2_OpenFeedHeat_Test3"

dymola = DymolaInterface()

pathsfile = open(str(ALPACApath)+'/Utilities/ModelicaPaths.txt', 'r')
Paths = pathsfile.readlines()
for path in Paths:
    path = path.strip()
    dymola.openModel(path)

#Add any package dependencies to the enviroment and change working directory
dymola.openModel(str(ALPACApath)+"/ModelicaFiles/ControlTests.mo")
wd = 'Modelica.Utilities.System.setWorkDirectory("' + str(ALPACApath) + '\DymolaRunData")'
print(wd)
dymola.ExecuteCommand(wd) 


def getOrigResults():

    Orig_results = {}
    Orig_variables = ["Time","sensor_pT.T"]    
    for key in Orig_variables:
        Orig_results[key] = []
    Orig_trajsize = dymola.readTrajectorySize(str(ALPACApath)+"/RunData/Original_Temp_Profile.mat")
    Orig_signals = dymola.readTrajectory(str(ALPACApath)+"/RunData/Original_Temp_Profile.mat", Orig_variables, Orig_trajsize)
    
    for i in range(0,len(Orig_variables),1):
        Orig_results[Orig_variables[i]].extend(Orig_signals[i])
        
    for i in range(0,len(Orig_results["Time"]),1):
        Orig_results["Time"][i] = Orig_results["Time"][i] - 9900
        
    return Orig_results

def getNewResults():

    New_results = {}
    New_variables = ["Time","sensor_pT.T"]    
    for key in New_variables:
        New_results[key] = []
    New_trajsize = dymola.readTrajectorySize(str(ALPACApath)+"/DymolaRunData/SteamTurbine_L2_OpenFeedHeat_Test3.mat")
    New_signals = dymola.readTrajectory(str(ALPACApath)+"/DymolaRunData/SteamTurbine_L2_OpenFeedHeat_Test3.mat", New_variables, New_trajsize)
    
    for i in range(0,len(New_variables),1):
        New_results[New_variables[i]].extend(New_signals[i])
        
    for i in range(0,len(New_results["Time"]),1):
        New_results["Time"][i] = New_results["Time"][i] - 9900
        
    return New_results



def RunDymola(FeedForward):
    dymola.writeTrajectory(str(ALPACApath)+"/DymolaRunData/feedforward.mat", ["Time","Feed"], FeedForward)
    
    #reopen model in new working directory
    dymola.openModel(model)
    
    #translate the model and create dymosym files
    dymola.translateModel(model)
    
    dymola.ExecuteCommand('importInitial("'+ str(ALPACApath) + '/RunData/Starting_9900New.txt")')
    
    #Initialise the model and create first dsfinal file
    result = dymola.simulateModel(model, startTime=9900, stopTime = 10500, outputInterval=1, method="Esdirk45a",resultFile="SteamTurbine_L2_OpenFeedHeat_Test3")
    
    #error out if initial simulation fails
    if not result:
        print("Simulation failed. Below is the translation log.")
        log = dymola.getLastErrorLog()
        print(log)
        dymola.exit(1)
    
    return result
        
origResults = getOrigResults()


def funct(consts):
    x = np.arange(0,500,0.1)
    FF = []

    for point in x:
        FFp = 0
        for i in range(0,len(consts)-2,1):
            FFp += consts[i]*jv(i,point/consts[len(consts)-2]) + consts[len(consts)-1]
        FF.append(FFp)
    
    x = x + 10000
    x = np.insert(x,0,0)
    x = np.insert(x,1,10000)
    xt = np.atleast_2d(x)
    xt = np.transpose(xt)
    
    FF.insert(0,0)
    FF.insert(0,0)
    FF = np.array(FF)
    FFt = np.atleast_2d(FF)
    FFt = np.transpose(FFt)
    
    Tup = np.append(xt,FFt, axis =1)
        
    FeedForward = Tup.tolist()
    RunDymola(FeedForward)
    newResults = getNewResults()
    
    result = 0
    for i in range(len(newResults["Time"])-1):
        t_diff = newResults["Time"][i+1]-newResults["Time"][i]
        result += (t_diff/2)*abs(newResults["sensor_pT.T"][i]+ newResults["sensor_pT.T"][i+1] - 2*673)
        
    print(result)
    print(consts)
    fig, axs = plt.subplots(2,1)
    axs[0].plot(origResults["Time"],origResults["sensor_pT.T"], label = 'Original')
    axs[0].plot(newResults["Time"],newResults["sensor_pT.T"], label = 'New')
    axs[0].axhspan(671.15, 675.15, color='green', alpha=0.5, label = "Temperature Band")
    axs[0].set_xlim(-15,550)
    #axs.set_xlim(0,200)
    axs[0].legend()
    axs[0].set_title('Temperature')
    # axs[0, 1].plot(t1, mdots, 'tab:orange')
    # axs[0, 1].set_title('Pump Mdot')
    axs[1].plot(Tup[:,0]-9900,Tup[:,1], 'tab:green')
    axs[1].set_xlim(-15,550)
    axs[1].set_title('FF Signal')
    # axs[1, 1].plot(t1, TCVdps, 'tab:red')
    # axs[1, 1].set_title('TCV pressure drop')
    fig.tight_layout()
    axs[1].set(xlabel='Time / s')
    plt.show()
    
    return result


consts = [1,1,1,100,-0.5]
res = minimize(funct, consts, method='Powell', tol=1e-2, bounds = [(-20,20),(-20,20),(-20,20),(0,200),(-0.7,0)])
print(res.x)

consts = res.x

x = np.arange(0,500,0.1)
FF = []

for point in x:
    FFp = 0
    for i in range(0,len(consts)-2,1):
        FFp += consts[i]*jv(i,point/consts[len(consts)-2]) + consts[len(consts)-1]
    FF.append(FFp)

x = x + 10000
x = np.insert(x,0,0)
x = np.insert(x,1,10000)
xt = np.atleast_2d(x)
xt = np.transpose(xt)

FF.insert(0,0)
FF.insert(0,0)
FF = np.array(FF)
FFt = np.atleast_2d(FF)
FFt = np.transpose(FFt)

Tup = np.append(xt,FFt, axis =1)
    
FeedForward = Tup.tolist()
RunDymola(FeedForward)
newResults = getNewResults()

result = 0
for i in range(len(newResults["Time"])-1):
    t_diff = newResults["Time"][i+1]-newResults["Time"][i]
    result += (t_diff/2)*abs(newResults["sensor_pT.T"][i]+ newResults["sensor_pT.T"][i+1] - 2*673)

fig, axs = plt.subplots(2,1)
axs[0].plot(origResults["Time"],origResults["sensor_pT.T"], label = 'Original')
axs[0].plot(newResults["Time"],newResults["sensor_pT.T"], label = 'New')
axs[0].axhspan(671.15, 675.15, color='green', alpha=0.5, label = "Temperature Band")
axs[0].set_xlim(-15,550)
#axs.set_xlim(0,200)
axs[0].legend()
axs[0].set_title('Temperature')
# axs[0, 1].plot(t1, mdots, 'tab:orange')
# axs[0, 1].set_title('Pump Mdot')
axs[1].plot(Tup[:,0]-9900,Tup[:,1], 'tab:green')
axs[1].set_xlim(-15,550)
axs[1].set_title('FF Signal')
# axs[1, 1].plot(t1, TCVdps, 'tab:red')
# axs[1, 1].set_title('TCV pressure drop')
fig.tight_layout()
axs[1].set(xlabel='Time / s')