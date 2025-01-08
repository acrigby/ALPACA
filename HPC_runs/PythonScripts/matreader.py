import os

import torch
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch import nn
from csv import writer
import os
import logging
import math
import scipy.io
import pandas as pd
import numpy as np
import re

def relativeDiff(f1, f2):
  """
    Given two floats, safely compares them to determine relative difference.
    @ In, f1, float, first value (the value to compare to f2, "measured")
    @ In, f2, float, second value (the value being compared to, "actual")
    @ Out, relativeDiff, float, (safe) relative difference
  """
  if not isinstance(f1, float):
    try:
      f1 = float(f1)
    except ValueError:
      raise RuntimeError(f'Provided argument to compareFloats could not be cast as a float!  First argument is {f1} type {type(f1)}')
  if not isinstance(f2, float):
    try:
      f2 = float(f2)
    except ValueError:
      raise RuntimeError(f'Provided argument to compareFloats could not be cast as a float!  Second argument is {f2} type {type(f2)}')
  diff = abs(diffWithInfinites(f1, f2))
  # "scale" is the relative scaling factor
  scale = f2
  # protect against div 0
  if f2 == 0.0:
    # try using the "measured" for scale
    if f1 != 0.0:
      scale = f1
    # at this point, they're both equal to zero, so just divide by 1.0
    else:
      scale = 1.0
  if abs(scale) == np.inf:
    # no mathematical rigor here, but typical algorithmic use cases
    if diff == np.inf:
      return np.inf # assumption: inf/inf = 1
    else:
      return 0.0 # assumption: x/inf = 0 for all finite x

  return diff/abs(scale)

def compareFloats(f1, f2, tol=1e-6):
  """
    Given two floats, safely compares them to determine equality to provided relative tolerance.
    @ In, f1, float, first value (the value to compare to f2, "measured")
    @ In, f2, float, second value (the value being compared to, "actual")
    @ In, tol, float, optional, relative tolerance to determine match
    @ Out, compareFloats, bool, True if floats close enough else False
  """
  diff = relativeDiff(f1, f2)

  return diff < tol

def diffWithInfinites(a, b):
  """
    Calculates the difference a-b and treats infinites.  We consider infinites to have equal values, but
    inf - (- inf) = inf.
    @ In, a, float, first value (could be infinite)
    @ In, b, float, second value (could be infinite)
    @ Out, res, float, b-a (could be infinite)
  """
  if abs(a) == np.inf or abs(b) == np.inf:
    if a == b:
      res = 0 #not mathematically rigorous, but useful algorithmically
    elif a > b:
      res = np.inf
    else: # b > a
      res = -np.inf
  else:
    res = a - b

  return res


def matreader(matSourceFileName,variablesToLoad):
    _vars = {}
    _blocks = []
    _namesData1 = []
    _namesData2 = []
    _timeSeriesData1 = []
    _timeSeriesData2 = []

    mat = scipy.io.loadmat(matSourceFileName, chars_as_strings=False)

    #Define the functions that extract strings from the matrix:
    #  - strMatNormal: for parallel string
    #  - strMatTrans:  for vertical string
    # These functions join the strings together, resulting in one string in each row, and remove
    #   trailing whitespace.
    strMatNormal = lambda a: [''.join(s).rstrip() for s in a]
    strMatTrans  = lambda a: [''.join(s).rstrip() for s in zip(*a)]

    # Define the function that returns '1.0' with the sign of 'x'
    sign = lambda x: math.copysign(1.0, x)

    # Check the structure of the output file.
    try:
        fileInfo = strMatNormal(mat['Aclass'])
    except KeyError:
        raise Exception('File structure not supported!')

    # Check the version of the output file (version 1.1).
    #if fileInfo[1] == '1.1' and fileInfo[3] == 'binTrans':
    names = strMatTrans(mat['name']) # names
    descr = strMatTrans(mat['description']) # descriptions
    for i in range(len(names)):
        d = mat['dataInfo'][0][i] # data block
        x = mat['dataInfo'][1][i] # column (original)
        c = abs(x)-1  # column (reduced)
        s = sign(x)   # sign
        if c:
            _vars[names[i]] = (descr[i], d, c, float(s))
            if not d in _blocks:
                _blocks.append(d)
        else:
            _absc = (names[i], descr[i])

    # Extract the trajectory for the variable 'Time' and store the data in the variable 'timeSteps'.
    timeSteps = mat['data_2'][0]

    # Compute the number of output points of trajectory (time series data).
    numOutputPts = timeSteps.shape[0]

    # Convert the variable type of 'timeSteps' from '1-d array' to '2-d array'.
    timeStepsArray = np.array([timeSteps])

    # Extract the names and output points of all variables and store them in the variables:
    #  - _namesData1: Names of parameters
    #  - _namesData2: Names of the variables that are not parameters
    #  - _timeSeriesData1: Trajectories (time series data) of '_namesData1'
    #  - _timeSeriesData2: Trajectories (time series data) of '_namesData2'
    for (k,v) in _vars.items():
        readIt = True
        if len(variablesToLoad) > 0 and k not in variablesToLoad:
            readIt = False
        if readIt:
            dataValue = mat['data_%d' % (v[1])][v[2]]
            if v[3] < 0:
                dataValue = dataValue * -1.0
            if v[1] == 1:
                _namesData1.append(k)
                _timeSeriesData1.append(dataValue)
            elif v[1] == 2:
                _namesData2.append(k)
                _timeSeriesData2.append(dataValue)
            else:
                raise Exception('File structure not supported!')
    timeSeriesData1 = np.array(_timeSeriesData1)
    timeSeriesData2 = np.array(_timeSeriesData2)

    #print(timeSeriesData1,  timeSeriesData2)

    # The csv writer places quotes arround variables that contain a ',' in the name, i.e.
    # a, "b,c", d would represent 3 variables 1) a 2) b,c 3) d. The csv reader in RAVEN does not
    # suport this convention.
    # => replace ',' in variable names with '@', i.e.
    # a, "b,c", d will become a, b@c, d
    for mylist in [_namesData1, _namesData2]:
        for i in range(len(mylist)):
            if ',' in mylist[i]:
                mylist[i]  = mylist[i].replace(',', '@')

    # Recombine the names of the variables and insert the variable 'Time'.
    # Order of the variable names should be 'Time', _namesData1, _namesData2.
    # Also, convert the type of the resulting variable from 'list' to '2-d array'.
    varNames = np.array([[_absc[0]] + _namesData1 + _namesData2])

    # Compute the number of parameters.
    sizeParams = timeSeriesData1.shape[0]

    # Create a 2-d array whose size is 'the number of parameters' by 'number of ouput points of the trajectories'.
    # Fill each row in a 2-d array with the parameter value.
    Data1Array =  np.full((sizeParams,numOutputPts),1.)
    for n in range(sizeParams):
        Data1Array[n,:] = timeSeriesData1[n,0]

    # Create an array of trajectories, which are to be written to CSV file.
    varTrajectories = np.matrix.transpose(np.concatenate((timeStepsArray,Data1Array,timeSeriesData2), axis=0))
    # create output response dictionary
    t = pd.Series(varTrajectories[:,0])
    m = t.duplicated()
    if len(t[m]):
    # duplicated values
        tIndex = None
        iIndex = 1
        for i in range(len(t[m])):
            index = t[m].index[i]
            if tIndex is None:
                tIndex = t[index]
            else:
                if compareFloats(tIndex, t[index], tol=1.0E-15):
                    iIndex += 1
                else:
                    iIndex = 1
                    tIndex = t[index]
            t[index] = t[index] + np.finfo(float).eps*t[index]*iIndex
        varTrajectories[:,0] = t.to_numpy()
    response = {var:varTrajectories[:,i] for (i, var) in enumerate(varNames[0])}

    return response

def createNewInput(currentInputFiles, oriInputFiles, changedvars):
    """
      Generate a new Dymola input file (txt format) from the original, changing parameters
      as specified in Kwargs['SampledVars']. In addition, it creaes an additional input file including the vector data to be
      passed to Dymola.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
            where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    # Figure out the new file name and put it into the proper place in the return list
    #newInputFiles = copy.deepcopy(currentInputFiles)
    originalPath = oriInputFiles[0]
    #newPath = os.path.join(os.path.split(originalPath)[0], "DM" + Kwargs['prefix'] + os.path.split(originalPath)[1])
    #currentInputFiles[index].setAbsFile(newPath)
    # Define dictionary of parameters and pre-process the values.
    # Each key is a parameter name (including the full model path in Modelica_ dot notation) and
    #   each entry is a parameter value. The parameter name includes array indices (if any) in
    #   Modelica_ representation (1-based indexing). The values must be representable as scalar
    #   numbers (integer or floating point). *True* and *False* (not 'true' and 'false') are
    #   automatically mapped to 1 and 0. Enumerations must be given explicitly as the unsigned integer
    #   equivalent. Strings, functions, redeclarations, etc. are not supported.
    varDict = changedvars

    vectorsToPass= {}
    for key, value in list(varDict.items()):
      if isinstance(value, bool):
        varDict[key] = 1 if value else 0

    # Do the search and replace in input file "DymolaInitialisation"
    # Aliases for some regular sub-expressions.
    u = '\\d+' # Unsigned integer
    i = '[+-]?' + u # Integer
    f = i + '(?:\\.' + u + ')?(?:[Ee][+-]' + u + ')?' # Floating point number

    # Possible regular expressions for a parameter specification (with '%s' for
    #   the parameter name)
    patterns = [# Dymola 1- or 2-line parameter specification
                (r'(^\s*%s\s+)%s(\s+%s\s+%s\s+%s\s+%s\s*#\s*%s\s*$)'
                 % (i, f, f, f, u, u, '%s')),
                (r'(^\s*)' + i + r'(\s*#\s*%s)'),
                (r'(^\s*)' + f + r'(\s*#\s*%s)'),
                # From Dymola:
                # column 1: Type of initial value
                #           = -2: special case: for continuing simulation
                #                               (column 2 = value)
                #           = -1: fixed value   (column 2 = fixed value)
                #           =  0: free value, i.e., no restriction
                #                               (column 2 = initial value)
                #           >  0: desired value (column 1 = weight for
                #                                           optimization
                #                                column 2 = desired value)
                #                 use weight=1, since automatic scaling usually
                #                 leads to equally weighted terms
                # column 2: fixed, free or desired value according to column 1.
                # column 3: Minimum value (ignored, if Minimum >= Maximum).
                # column 4: Maximum value (ignored, if Minimum >= Maximum).
                #           Minimum and maximum restrict the search range in
                #           initial value calculation. They might also be used
                #           for scaling.
                # column 5: Category of variable.
                #           = 1: parameter.
                #           = 2: state.
                #           = 3: state derivative.
                #           = 4: output.
                #           = 5: input.
                #           = 6: auxiliary variable.
                # column 6: Data type of variable.
                #           = 0: real.
                #           = 1: boolean.
                #           = 2: integer.
               ]
    # These are tried in order until there is a match. The first group or pair
    #   of parentheses contains the text before the parameter value and the second
    #   contains the text after it (minus one space on both sides for clarity).

    # Read the file.
    with open(originalPath, 'r') as src:
      text = src.read()

    # Set the parameters.
    for name, value in varDict.items():
      # skip in the special key for the index mapper
      if name == '_indexMap':
        continue
      namere = re.escape(name) # Escape the dots, square brackets, etc.
      for pattern in patterns:
        text, n = re.subn(pattern % namere, r'\g<1>%s\2' % value, text, 1,
                          re.MULTILINE)
        if n == 1:
          break
      else:
        raise AssertionError(
          "Parameter %s does not exist or is not formatted as expected "
          "in %s." % (name, originalPath))

    # Re-write the file.
    with open(currentInputFiles[0], 'w') as src:
      src.write(text)

    return currentInputFiles