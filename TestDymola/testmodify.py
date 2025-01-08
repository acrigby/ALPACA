from matreader import matreader, createNewInput

import os 
import matplotlib.pyplot as plt

print(os.path.abspath(os.curdir))
ALPACApath = os.path.abspath(os.curdir)
os.chdir("TestDymola")

oriInputFiles = [str(ALPACApath) +'/TestDymola/' + "start.txt"]
currentInputFiles = [str(ALPACApath) +'/TestDymola/' + "modifyds.txt"]

changedvars = {'FeedForward.k':0} 

newinput = createNewInput(currentInputFiles, oriInputFiles, changedvars)

os.system('./dymosim ' + newinput[0])

FFlist = [0,-1,-2,3,2,1,0]
temps = []
times = []

for k in FFlist:

    changedvars = {'FeedForward.k':k} 
    oriInputFiles = [str(ALPACApath) +'/TestDymola/' + "dsfinal.txt"]

    newinput = createNewInput(currentInputFiles, oriInputFiles, changedvars)

    os.system('./dymosim ' + newinput[0])

    matSourceFileName = "dsres.mat"

    variablesToLoad = ["sensor_pT.T"]    

    response = matreader(matSourceFileName,variablesToLoad)

    temps.extend(response['sensor_pT.T'])
    times.extend(response['Time'])

plt.plot(times,temps)
print(str(ALPACApath) +'/TestDymola/mat.png')
plt.savefig(str(ALPACApath) +'/TestDymola/mat.png')
plt.close()
