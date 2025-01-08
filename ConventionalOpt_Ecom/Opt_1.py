# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:25:27 2023

@author: aidan
"""

import matplotlib
import matplotlib.pyplot as plt
from dymola.dymola_interface import DymolaInterface
from optimparallel import minimize_parallel
import numpy as np
import pandas as pd
import os
from scipy.special import jv
from scipy.optimize import minimize, basinhopping
print(os.path.abspath(os.curdir))
#os.chdir("..")
print(os.path.abspath(os.curdir))
ALPACApath = os.path.abspath(os.curdir)

f = open("./Utilities/ALPACAASCII.txt", 'r')
print(''.join([line for line in f]))



def getNewResults(dymola,tid):

    New_results = {}
    New_variables = ["Time","BOP.sensorW.W","PowerDemand.y[1]","DNI_Input.y[1]","CosEff_Input.y[1]", "BOP.deaerator.medium.p","dual_Pipe_CTES_Controlled_Feedwater.CTES.T_Ave_Conc","BOP.sensor_T2.T","SMR_Taveprogram.core.Q_total.y" ]
    for key in New_variables:
        New_results[key] = []
    New_trajsize = dymola.readTrajectorySize(str(ALPACApath)+"/ConventionalOpt_Ecom/"+str(tid)+"/dsres.mat")
    New_signals = dymola.readTrajectory(str(ALPACApath)+"/ConventionalOpt_Ecom/"+str(tid)+"/dsres.mat", New_variables, New_trajsize)
    
    for i in range(0,len(New_variables),1):
        New_results[New_variables[i]].extend(New_signals[i])
        
    for i in range(0,len(New_results["Time"]),1):
        New_results["Time"][i] = New_results["Time"][i] - 1e6
        
    return New_results



def RunDymola(FeedForward,tid,dymola,model):
    dymola.writeTrajectory(str(ALPACApath)+"/ConventionalOpt_Ecom/"+str(tid)+"/feedforward.mat", ["Time","Feed"], FeedForward)
    
    #reopen model in new working directory
    dymola.openModel(model)
    
    #translate the model and create dymosym files
    dymola.translateModel(model)
    
    dymola.ExecuteCommand('importInitial("'+ str(ALPACApath) + '/ConventionalOpt_Ecom/'+str(tid)+'/start2.txt")')
    
    #Initialise the model and create first dsfinal file
    result = dymola.simulateModel(model, startTime=1e6, stopTime = 1.036e6, outputInterval=10, method="Esdirk45a",resultFile="dsres")
    
    print(result)
    #error out if initial simulation fails
    if not result:
        print("Simulation failed. Below is the translation log.")
        log = dymola.getLastErrorLog()
        print(log)
        dymola.exit(1)
    
    return result


def funct(consts,price_schedule,prices):
    
    tid = os.getpid()

    print(tid)

    model = "NHES.Systems.Examples.FWHPT.LWR_L2_Turbine_AdditionalFeedheater_NewControl2_Deaerator_2_Opt2"

    # Instantiate the Dymola interface and start Dymola
    dymola = DymolaInterface(dymolapath="/apps/local/dymola/2023.3/bin64/dymola")
    print(dymola.DymolaVersion())

    dymola.openModel(str(ALPACApath)+"/ModelicaFiles/ControlTests.mo")
    dymola.openModel("/home/rigbac/Projects/HYBRID/TRANSFORM-Library/TRANSFORM/package.mo")
    dymola.openModel("/home/rigbac/Projects/HYBRID/Models/NHES/package.mo")
    dymola.openModel("/home/rigbac/Projects/Thermocycle-library/ThermoCycle/package.mo")
    dymola.openModel("/home/rigbac/Projects/ExternalMedia/Modelica/ExternalMedia/package.mo")

    os.system('cp -a /home/rigbac/Projects/ALPACA/ConventionalOpt_Ecom/DymolaFiles/. /home/rigbac/Projects/ALPACA/ConventionalOpt_Ecom/' + str(tid))
    wd = 'Modelica.Utilities.System.setWorkDirectory("' + str(ALPACApath) + '/ConventionalOpt_Ecom/' + str(tid) + '")'

    print(wd)

    dymola.ExecuteCommand(wd) 


    #reopen model in new working directory
    dymola.openModel(model)

    #translate the model and create dymosym files
    dymola.translateModel(model)

    P=36000
    N = len(consts)

    x = np.arange(3600,P,(39600)/(N))

    Power = []

    for i in range(0,len(x),1):
        FFp = consts[i]*8e6 + 48.6e6
        Power.append(FFp)
    
    for i in range(len(x)):
        x = np.insert(x,2*i+1 , x[2*i] + 2000)
        Power.insert(2*i+1,Power[2*i])

    x = np.append(x,2e6)
    Power.insert(0,48.6e6)
    Power.insert(1,48.6e6)
    Power.append(48.6e6)


    x = x + 1e6
    x = np.insert(x,0,0)
    x = np.insert(x,1,1000000)
    xt = np.atleast_2d(x)
    xt = np.transpose(xt)


    Power = np.array(Power)
    Powert = np.atleast_2d(Power)
    Powert = np.transpose(Powert)

    Tup = np.append(xt,Powert, axis =1)

        
    FeedForward = Tup.tolist()
    #print(FeedForward)
    RunDymola(FeedForward,tid,dymola,model)
    newResults = getNewResults(dymola,tid)

    print(price_schedule,prices)
    
    result = 0
    for i in range(len(newResults["Time"])-1):
        t = newResults["Time"][i]
        for i in range(len(price_schedule)):
            if price_schedule[i] <= t < price_schedule[i+1]:
                t_diff = newResults["Time"][i+1]-newResults["Time"][i]
                reward_price_t = prices[i]
                result += -(t_diff/(2*3600))*(newResults["BOP.sensorW.W"][i]+ newResults["BOP.sensorW.W"][i+1])*reward_price_t/1e6
    
    result = result + 10000*(newResults["dual_Pipe_CTES_Controlled_Feedwater.CTES.T_Ave_Conc"][-1]-273 - 180.65)
    print(result)
    print(consts)
    # Reset the score. The final score will be the total amount of steps before the pole falls
    t = 1003600
    powers = []
    price_list = []
    demands = []
    times = []
    scorelist = [0]
    tscore = [t]

    terminated = False
    truncated = False


    #price_schedule,prices = importdict('/home/rigbac/Projects/ALPACA/HPC_runs/Input/20240522-20240528 CAISO Day-Ahead Price','SCE',0) #start the function with the name of the file

    #"BOP.sensorW.W","PowerDemand.y","DNI_Input.y[1]","CosEff_Input.y[1]", "BOP.deaerator.medium.p","dual_Pipe_CTES_Controlled_Feedwater.CTES.T_Ave_Conc","BOP.sensor_T2.T","SMR_Taveprogram.core.Q_total.y" 

    pressures = []

    for u in range(len(newResults["BOP.deaerator.medium.p"])):
        pressures.append(newResults["BOP.deaerator.medium.p"][u]/101000)


    conctemps = []

    for c in range(len(newResults["dual_Pipe_CTES_Controlled_Feedwater.CTES.T_Ave_Conc"])):
        conctemps.append(newResults["dual_Pipe_CTES_Controlled_Feedwater.CTES.T_Ave_Conc"][c]-273)

    feedtemps = []

    for f in range(len(newResults["BOP.sensor_T2.T"])):
        feedtemps.append(newResults["BOP.sensor_T2.T"][f]-273)

    dnis = []

    for d in range(len(newResults["DNI_Input.y[1]"])):
        dnis.append(newResults["DNI_Input.y[1]"][d])

    ts = []

    for ti in range(len(newResults["Time"])):
        ts.append(newResults["Time"][ti]/ (3600*24))


    fig, axs = plt.subplots(2,3, figsize=(20,12))

    
    fig.suptitle(f"Profit = ${result}")
    axs[0,0].plot(ts,newResults["PowerDemand.y[1]"], label = 'Demand')
    axs[0,0].set_title('Demands')
    axs[1,0].set(ylabel='Demand / MW')

    #axs[0].set_xlim(0,250)
    #axs[0].legend()
    #axs[0].set_title('Temperature')
    # axs[0, 1].plot(t1, mdots, 'tab:orange')
    # axs[0, 1].set_title('Pump Mdot')
    axs[0,0].plot(ts,newResults["BOP.sensorW.W"], label = 'Power')
    axs[1,0].set_title('Prices')
    axs[1,0].set(ylabel='Price/ $/MWh')
    axs[1,0].plot(times, price_list, 'tab:red')

    axs[1,1].set_title('Average Concrete Temperature')
    axs[1,1].plot(ts, conctemps, 'tab:red')
    axs[1,1].set(ylabel='Temperature / $\circ$ C')

    axs[0,1].set_title('Deaerator Pressure')
    axs[0,1].plot(ts, pressures, 'tab:red')
    axs[0,1].set(ylabel='Pressure / bar')
    axs[0,1].axhspan(1, 1.6, color='green', alpha=0.5, label = "Pressure Band")

    #axs[0,2].set_title('Profit')
    #axs[0,2].plot(tscore, scorelist, 'tab:red')
    #axs[0,2].set(ylabel='Profit / $')

    axs[0,2].set_title('DNI')
    axs[0,2].plot(ts, dnis, 'tab:red')
    axs[0,2].set(ylabel='DNI / MW/m^2')

    axs[1,2].set_title('Feedwater Temperature')
    axs[1,2].plot(ts, feedtemps, 'tab:red')
    axs[1,2].set(ylabel='Temperature / $\circ$ C')
    axs[1,2].axhspan(147.5, 148.5, color='green', alpha=0.5, label = "Pressure Band")

    # axs[1, 1].set_title('TCV pressure drop')
    fig.tight_layout()
    axs[0,0].set(xlabel='Time / days')
    axs[1,0].set(xlabel='Time / days')
    axs[0,1].set(xlabel='Time / days')
    axs[1,1].set(xlabel='Time / days')
    axs[0,2].set(xlabel='Time / days')
    axs[1,2].set(xlabel='Time / days')
        
    #display.display(plt.gcf())
    plt.savefig(str(ALPACApath)+"/ConventionalOpt_Ecom/"+str(tid)+"/Iteration.png")
    plt.close()

    dymola.exit(1)
    
    return result

def importdict3(filename, offset, fileyear):#creates a function to read the csv
    times_seconds = []
    prices = []
    DNIs = []
    IAMs = []
    #create data frame from csv with pandas module
    df=pd.read_csv(filename+str(fileyear)+'.csv', names=['Year', 'Month', 'Day', 'Hour', 'Minute', 'DNI', 'Price', 'IAM'],sep=',',skiprows=1) #or:, infer_datetime_format=True)
    fileDATES=df.T.to_dict().values()#export the data frame to a python dictionary
    fileDATES = list(fileDATES)
    shortdict = fileDATES[offset:offset+103]
    j=0
    for line in shortdict:
        times_seconds.append(j*3600)
        prices.append(line['Price'])
        DNIs.append(line['DNI'])
        IAMs.append(line['IAM'])
        j = j + 1

    return times_seconds, prices, DNIs, IAMs #return list of times in seconds and prices

price_schedule,prices, DNIs, IAMs = importdict3('/home/rigbac/Projects/ALPACA/HPC_runs/Input/TestSyn',0,1) #start the function with the name of the file

consts = np.ones(10)*0.5
n = 0
P=363600 
N = len(consts)


DNIs.insert(0,0)
DNIs.append(0)
DNIs= np.array(DNIs)
DNIst = np.atleast_2d(DNIs)
DNIst = np.transpose(DNIst)


IAMs.insert(0,0)
IAMs.append(0)
IAMs= np.array(IAMs)
IAMst = np.atleast_2d(IAMs)
IAMst = np.transpose(IAMst)

price_schedule = np.array(price_schedule)
price_schedule = np.append(price_schedule,2e6)
price_schedule = price_schedule + 1e6
price_schedule = np.insert(price_schedule,0,0)
xt = np.atleast_2d(price_schedule)
xt = np.transpose(xt)

Tup_DNI = np.append(xt,DNIst, axis =1)

Tup_IAM = np.append(xt,IAMst, axis =1)
    
DNI = Tup_DNI.tolist()

print(DNI)

IAM = Tup_IAM.tolist()

print(IAM)

#dymola.writeTrajectory(str(ALPACApath)+"/ConventionalOpt_Ecom/DNI.mat", ["Time","DNI"], DNI)
#dymola.writeTrajectory(str(ALPACApath)+"/ConventionalOpt_Ecom/CosEff.mat", ["Time","CosEFF"], IAM)

bnds = ((0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1),
(0,1), (0,1))


res = minimize_parallel(funct, consts,args=(price_schedule,prices),tol=0.01, bounds=bnds, parallel={'max_workers':5})
print(res.x)

consts = res.x