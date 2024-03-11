# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:38:03 2023

@author: localuser
"""

import scipy.io
import matplotlib
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('SteamTurbine_L2_OpenFeedHeat_Test_6.mat')

time = mat['data'][:,0]


powerdemand = mat['data'][:,1]
power = mat['data'][:,2]
mass_flow = mat['data'][:,3]
Temp = mat['data'][:,4]
deaer_level = mat['data'][:,5]


#plt.plot(time,[power, powerdemand])
fig, axs = plt.subplots(2,2 , sharex=True)
axs[0,0].plot(time,powerdemand/1000000, label = 'Electrical Power Demand')
axs[0,0].plot(time,power/1000000, label = 'Electrical Power')
#axs[0,0].axhspan(671.15, 675.15, color='green', alpha=0.5, label = "Temperature Band")
axs[0,0].set_xlim(9000,12000)
plt.setp(axs[0, 0], ylabel='$W_{Elect}$ / MW')
#axs[0,0].legend()
#axs[0,0].set_title('Power')
# Shrink current axis's height by 10% on the bottom
box = axs[0,0].get_position()
axs[0,0].set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
axs[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=1, fontsize="5")


axs[0,1].plot(time, mass_flow, 'tab:green')
plt.setp(axs[0, 1], ylabel='$\dot m_{FWCP}$ / kg/s')
axs[0,1].set_xlim(9000,13000)
axs[0,1].set_ylim(58,59)


fig.tight_layout()
plt.setp(axs[1, :], xlabel='Time / s')

plt.show()
    
plt.close()
#plt.plot(time,hotTemp)