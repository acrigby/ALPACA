<<<<<<< HEAD:ME759/Test.py
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:19:01 2023

@author: aidan
"""

from subprocess import Popen, PIPE

s_augmented = [0,1,2,3,4]

dsdt = Popen(['./rk4 %s %s %s %s %s' %(str(s_augmented[0]),str(s_augmented[1]),str(s_augmented[2]),str(s_augmented[3]),str(s_augmented[4]))], shell=True, stdout=PIPE, stdin=PIPE).communicate()[0]

dsdt = dsdt.decode('utf-8')

print(type(dsdt))

dsdt = dsdt.split(',')

output = [float(n) for n in dsdt]

print(output)
=======
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:19:01 2023

@author: aidan
"""

from subprocess import Popen, PIPE

<<<<<<< HEAD
p = Popen('Subprocess.exe', shell=True, stdout=PIPE, stdin=PIPE)
for ii in range(10):
    value = str(ii) + '\n'
    value = bytes(value, 'UTF-8')  # Needed in Python 3.
    p.stdin.write(value)
    p.stdin.flush()
    result = p.stdout.readline().strip()
    print(result)
=======
s_augmented = [0,1,2,3,4]

dsdt = Popen(['./rk4 %s %s %s %s %s' %(str(s_augmented[0]),str(s_augmented[1]),str(s_augmented[2]),str(s_augmented[3]),str(s_augmented[4]))], shell=True, stdout=PIPE, stdin=PIPE).communicate()[0]

dsdt = dsdt.decode('utf-8')

print(type(dsdt))

dsdt = dsdt.split(',')

output = [float(n) for n in dsdt]

print(output)
>>>>>>> 81a0513285b5edfbd798a971f3ca0ad6c051e1d8
>>>>>>> b995bb8dd986855b990cf174bb41d2efb36ea0c7:ME759/TestExternalC.py
