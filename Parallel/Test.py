# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 13:07:10 2023

@author: localuser
"""

import pathlib
import os

# src: The directory where this script is
src = pathlib.Path(__file__).parent

target = pathlib.Path( str(src) + f"/tmp/dir{1}")

os.makedirs(target)