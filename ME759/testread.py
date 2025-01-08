from matreader import matreader

import os 
import matplotlib.pyplot as plt
print(os.path.abspath(os.curdir))
ALPACApath = os.path.abspath(os.curdir)
os.chdir("RunData")

matSourceFileName = "Original_Temp_Profile.mat"

variablesToLoad = ["sensor_pT.T"]    

response = matreader(matSourceFileName,variablesToLoad)

plt.plot(response['Time'],response['sensor_pT.T'])
print(str(ALPACApath) +'/ME759/mat.png')
plt.savefig(str(ALPACApath) +'/ME759/mat1.png')
plt.close()