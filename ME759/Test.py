# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:19:01 2023

@author: aidan
"""

from subprocess import Popen, PIPE

p = Popen('Subprocess.exe', shell=True, stdout=PIPE, stdin=PIPE)
for ii in range(10):
    value = str(ii) + '\n'
    value = bytes(value, 'UTF-8')  # Needed in Python 3.
    p.stdin.write(value)
    p.stdin.flush()
    result = p.stdout.readline().strip()
    print(result)
