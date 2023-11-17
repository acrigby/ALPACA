# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:19:01 2023

@author: aidan
"""

import subprocess

s = subprocess.check_output(["echo", "Hello World!"])

print(s)
