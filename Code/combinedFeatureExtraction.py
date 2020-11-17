# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:10:41 2019

@author: OwensLab

Description: Pulls features like maximum voltage, minimum voltage, maximum frequency,
etc. from the raw MES taken form the DAQ into a single array.
"""

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import os

# Location of input files
PATH = 'C://Users/OwensLab/Documents/Preston/data/PaytonData'

# Splits string into characters 
def split(word): 
    return list(word)

Vmax = []
Vmin = []
Vrange = []
Vsum = []
Vavg = []
Fmax = []
Fmin = []
Fsum = []
Favg = []
Frange = []
Name = []

# Extracts features such as Vmax, Vmin, etc.
def featureExtraction(data, outputFile, name, index):
     
     y = data['CHANNEL1']
     fs = 1000
     
     Name.append(name)
     
     #Vmax
     Vmax.append(y.max())
     #Vmin 
     Vmin.append(y.min())
     
     #Return the sum of voltages
     Vsum.append(y.sum())
     #print('sum =', Vsum)
     Vavg.append(y.mean())
     #print('Vavg =', Vavg)
     Vrange.append(Vmax[index] - Vmin[index])
     
     #find the Fourier transform of the data
     z = np.fft.rfft(y)
     ##get the corresponding frequencies
     f = np.fft.rfftfreq(np.size(y),1/fs)
     ##find the power spectrum
     P = np.abs(z)**2
     z = pd.Series(z)
     n = f > 3
     flow = f[n]
     Plow = P[n]
     Fmax.append(flow[np.argmax(Plow)])

     np.savez(outputFile, Name=Name, Vmax=Vmax, Vmin=Vmin, Vsum=Vsum, Vavg=Vavg, Vrange=Vrange, Fmax=Fmax)


#load and feature extract data from a given path
def loadData(PATH, ):
    index = 0 # makes sure that data is not overwritten in pickle
    name = 0
    #Read the data for each day
    for file in os.listdir(PATH):
        newPATH = os.path.join(PATH, file)
        # Read the data for each dataset in each day's folder: fist, null, etc.
        for file in os.listdir(newPATH):
            loadFile = os.path.join(newPATH, file)
            #print(loadFile)
            data = pd.read_csv(loadFile, skiprows=7)
            data.columns = ['Sample', 'Date', 'CHANNEL1', 'Events']
            folderName = file.split('.')[0]
            #print(folderName)
            if (folderName == 'null'):
                name=0
            elif (folderName == 'fist'):
                name=1
            elif (folderName == 'thumb'):
                name=2
            elif (folderName == 'index'):
                name=3
            elif (folderName == 'middle'):
                name=4
            elif (folderName == 'ring'):
                name=5
            else:
                name=6
            # location of output file
            PATH2 = 'C://Users/OwensLab/Documents/Preston/results/'
            outputFile = os.path.join(PATH2, 'PaytonResults/outputData.npz')
            featureExtraction(data, outputFile, name, index)
            index += 1
            
loadData(PATH)