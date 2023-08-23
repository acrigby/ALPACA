# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:44:04 2023

@author: aidan
"""
import os

print(os.path.abspath(os.curdir))
os.chdir("..")
print(os.path.abspath(os.curdir))

# path = Path("ALPACA/Utilities/ALPACAASCII.txt").parents[1]
# print(path.absolute())
f = open("./Utilities/ALPACAASCII.txt", 'r')
print(''.join([line for line in f]))