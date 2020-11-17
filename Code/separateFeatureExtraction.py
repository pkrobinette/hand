# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:20:55 2019

@author: OwensLab

Description: Pulls features like maximum voltage, minimum voltage, maximum frequency,
etc. from the raw MES taken form the DAQ into separate arrays.
"""

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import os

# Location of input files
PATH = 'C://Users/OwensLab/Documents/Preston/data/'
# Location of output file
PATH2 = 'C://Users/OwensLab/Documents/Preston/outputData.txt'
outputFile = open(PATH2, 'w+')
#outputFile.write("Check Check")
#outputFile.close()

# Splits string into characters 
def split(word): 
    return list(word)

Vmaxf=[]
Vminf = []
Vrangef = []
Vsumf = []
Vavgf = []
Fmaxf = []
Fminf = []
Fsumf = []
Favgf = []
Frangef = []

Vmaxn=[]
Vminn = []
Vrangen = []
Vsumn = []
Vavgn = []
Fmaxn = []
Fminn = []
Fsumn = []
Favgn = []
Frangen = []

Vmaxt = []
Vmint = []
Vranget = []
Vsumt = []
Vavgt = []
Fmaxt = []
Fmint = []
Fsumt = []
Favgt = []
Franget = []

Vmaxi = []
Vmini = []
Vrangei = []
Vsumi = []
Vavgi = []
Fmaxi = []
Fmini = []
Fsumi = []
Favgi = []
Frangei = []

Vmaxm = []
Vminm = []
Vrangem = []
Vsumm = []
Vavgm = []
Fmaxm = []
Fminm = []
Fsumm = []
Favgm = []
Frangem = []

Vmaxr = []
Vminr = []
Vranger = []
Vsumr = []
Vavgr = []
Fmaxr = []
Fminr = []
Fsumr = []
Favgr = []
Franger = []

Vmaxp = []
Vminp = []
Vrangep = []
Vsump = []
Vavgp = []
Fmaxp = []
Fminp = []
Fsump = []
Favgp = []
Frangep = []


# Extracts features such as Vmax, Vmin, etc.
def pinkieExtraction(data, outputFile, index):
     
     y = data['CHANNEL1']
        
     #Vmax
     Vmaxp.append(y.max())
     #Vmin 
     Vminp.append(y.min())
     
     #Return the sum of voltages
     Vsump.append(y.sum())
     #print('sum =', Vsum)
     Vavgp.append(y.mean())
     #print('Vavg =', Vavg)
     Vrangep.append(Vmaxp[index] - Vminp[index])
     
     #find the Fourier transform of the data
     z = np.fft.rfft(y)
     ##get the corresponding frequencies
     #f = np.fft.rfftfreq(np.size(y),1/fs)
     ##find the power spectrum
     #P = np.abs(z)**2
     z = pd.Series(z)
     #print(type(z))
     
     #Fmax
     Fmaxp.append(z.max())
     #Fmin 
     Fminp.append(z.min())
     #print('Fmax =', Fmax, '\nFmin =', Fmin)
     
     #Return the sum of frequencies
     Fsump.append(z.sum())
     #print('sum =', Fsum)
     #Average
     Favgp.append(z.mean())
     #print('Favg =', Favg)
     Frangep.append(Fmaxp[index] - Fminp[index])
     np.savez(outputFile, Vmax=Vmaxp, Vmin=Vminp, Vsum=Vsump, Vavg=Vavgp, Vrange=Vrangep, Fmax=Fmaxp, Fmin=Fminp, Fsum=Fsump, Favg=Favgp, Frange=Frangep)
     
     #plt.figure(2)
#    plt.clf()
#    plt.title('Frequency Space')
#    plt.loglog(f[2:],P[2:],'.-')
#    plt.show()
#
#    plt.figure(3)
#    plt.clf()
#    plt.loglog(Pnew[5:])
#    plt.show()
# Extracts features such as Vmax, Vmin, etc.
def ringExtraction(data, outputFile, index):
     
     y = data['CHANNEL1']
        
     #Vmax
     Vmaxr.append(y.max())
     #Vmin 
     Vminr.append(y.min())
     
     #Return the sum of voltages
     Vsumr.append(y.sum())
     #print('sum =', Vsum)
     Vavgr.append(y.mean())
     #print('Vavg =', Vavg)
     Vranger.append(Vmaxr[index] - Vminr[index])
     
     #find the Fourier transform of the data
     z = np.fft.rfft(y)
     ##get the corresponding frequencies
     #f = np.fft.rfftfreq(np.size(y),1/fs)
     ##find the power spectrum
     #P = np.abs(z)**2
     z = pd.Series(z)
     #print(type(z))
     
     #Fmax
     Fmaxr.append(z.max())
     #Fmin 
     Fminr.append(z.min())
     #print('Fmax =', Fmax, '\nFmin =', Fmin)
     
     #Return the sum of frequencies
     Fsumr.append(z.sum())
     #print('sum =', Fsum)
     #Average
     Favgr.append(z.mean())
     #print('Favg =', Favg)
     Franger.append(Fmaxr[index] - Fminr[index])
     np.savez(outputFile, Vmax=Vmaxr, Vmin=Vminr, Vsum=Vsumr, Vavg=Vavgr, Vrange=Vranger, Fmax=Fmaxp, Fmin=Fminr, Fsum=Fsumr, Favg=Favgr, Frange=Franger)
     
def middleExtraction(data, outputFile, index):
     
     y = data['CHANNEL1']
        
     #Vmax
     Vmaxm.append(y.max())
     #Vmin 
     Vminm.append(y.min())
     
     #Return the sum of voltages
     Vsumm.append(y.sum())
     #print('sum =', Vsum)
     Vavgm.append(y.mean())
     #print('Vavg =', Vavg)
     Vrangem.append(Vmaxm[index] - Vminm[index])
     
     #find the Fourier transform of the data
     z = np.fft.rfft(y)
     ##get the corresponding frequencies
     #f = np.fft.rfftfreq(np.size(y),1/fs)
     ##find the power spectrum
     #P = np.abs(z)**2
     z = pd.Series(z)
     #print(type(z))
     
     #Fmax
     Fmaxm.append(z.max())
     #Fmin 
     Fminm.append(z.min())
     #print('Fmax =', Fmax, '\nFmin =', Fmin)
     
     #Return the sum of frequencies
     Fsumm.append(z.sum())
     #print('sum =', Fsum)
     #Average
     Favgm.append(z.mean())
     #print('Favg =', Favg)
     Frangem.append(Fmaxm[index] - Fminm[index])
     np.savez(outputFile, Vmax=Vmaxm, Vmin=Vminm, Vsum=Vsumm, Vavg=Vavgm, Vrange=Vrangem, Fmax=Fmaxm, Fmin=Fminm, Fsum=Fsumm, Favg=Favgm, Frange=Frangem)
     
def indexExtraction(data, outputFile, index):
     
     y = data['CHANNEL1']
        
     #Vmax
     Vmaxi.append(y.max())
     #Vmin 
     Vmini.append(y.min())
     
     #Return the sum of voltages
     Vsumi.append(y.sum())
     #print('sum =', Vsum)
     Vavgi.append(y.mean())
     #print('Vavg =', Vavg)
     Vrangei.append(Vmaxi[index] - Vmini[index])
     
     #find the Fourier transform of the data
     z = np.fft.rfft(y)
     ##get the corresponding frequencies
     #f = np.fft.rfftfreq(np.size(y),1/fs)
     ##find the power spectrum
     #P = np.abs(z)**2
     z = pd.Series(z)
     #print(type(z))
     
     #Fmax
     Fmaxi.append(z.max())
     #Fmin 
     Fmini.append(z.min())
     #print('Fmax =', Fmax, '\nFmin =', Fmin)
     
     #Return the sum of frequencies
     Fsumi.append(z.sum())
     #print('sum =', Fsum)
     #Average
     Favgi.append(z.mean())
     #print('Favg =', Favg)
     Frangei.append(Fmaxi[index] - Fmini[index])
     np.savez(outputFile, Vmax=Vmaxi, Vmin=Vmini, Vsum=Vsumi, Vavg=Vavgi, Vrange=Vrangei, Fmax=Fmaxi, Fmin=Fmini, Fsum=Fsumi, Favg=Favgi, Frange=Frangei)
     
def thumbExtraction(data, outputFile, index):
     
     y = data['CHANNEL1']
        
     #Vmax
     Vmaxt.append(y.max())
     #Vmin 
     Vmint.append(y.min())
     
     #Return the sum of voltages
     Vsumt.append(y.sum())
     #print('sum =', Vsum)
     Vavgt.append(y.mean())
     #print('Vavg =', Vavg)
     Vranget.append(Vmaxt[index] - Vmint[index])
     
     #find the Fourier transform of the data
     z = np.fft.rfft(y)
     ##get the corresponding frequencies
     #f = np.fft.rfftfreq(np.size(y),1/fs)
     ##find the power spectrum
     #P = np.abs(z)**2
     z = pd.Series(z)
     #print(type(z))
     
     #Fmax
     Fmaxt.append(z.max())
     #Fmin 
     Fmint.append(z.min())
     #print('Fmax =', Fmax, '\nFmin =', Fmin)
     
     #Return the sum of frequencies
     Fsumt.append(z.sum())
     #print('sum =', Fsum)
     #Average
     Favgt.append(z.mean())
     #print('Favg =', Favg)
     Franget.append(Fmaxt[index] - Fmint[index])
     np.savez(outputFile, Vmax=Vmaxt, Vmin=Vmint, Vsum=Vsumt, Vavg=Vavgt, Vrange=Vranget, Fmax=Fmaxt, Fmin=Fmint, Fsum=Fsumt, Favg=Favgt, Frange=Franget)
     
def nullExtraction(data, outputFile, index):
     
     y = data['CHANNEL1']
        
     #Vmax
     Vmaxn.append(y.max())
     #Vmin 
     Vminn.append(y.min())
     
     #Return the sum of voltages
     Vsumn.append(y.sum())
     #print('sum =', Vsum)
     Vavgn.append(y.mean())
     #print('Vavg =', Vavg)
     Vrangen.append(Vmaxn[index] - Vminn[index])
     
     #find the Fourier transform of the data
     z = np.fft.rfft(y)
     ##get the corresponding frequencies
     #f = np.fft.rfftfreq(np.size(y),1/fs)
     ##find the power spectrum
     #P = np.abs(z)**2
     z = pd.Series(z)
     #print(type(z))
     
     #Fmax
     Fmaxn.append(z.max())
     #Fmin 
     Fminn.append(z.min())
     #print('Fmax =', Fmax, '\nFmin =', Fmin)
     
     #Return the sum of frequencies
     Fsumn.append(z.sum())
     #print('sum =', Fsum)
     #Average
     Favgn.append(z.mean())
     #print('Favg =', Favg)
     Frangen.append(Fmaxn[index] - Fminn[index])
     np.savez(outputFile, Vmax=Vmaxn, Vmin=Vminn, Vsum=Vsumn, Vavg=Vavgn, Vrange=Vrangen, Fmax=Fmaxn, Fmin=Fminn, Fsum=Fsumn, Favg=Favgn, Frange=Frangen)
     
def fistExtraction(data, outputFile, index):
     
     y = data['CHANNEL1']
        
     #Vmax
     Vmaxf.append(y.max())
     #Vmin 
     Vminf.append(y.min())
     
     #Return the sum of voltages
     Vsumf.append(y.sum())
     #print('sum =', Vsum)
     Vavgf.append(y.mean())
     #print('Vavg =', Vavg)
     Vrangef.append(Vmaxf[index] - Vminf[index])
     
     #find the Fourier transform of the data
     z = np.fft.rfft(y)
     ##get the corresponding frequencies
     #f = np.fft.rfftfreq(np.size(y),1/fs)
     ##find the power spectrum
     #P = np.abs(z)**2
     z = pd.Series(z)
     #print(type(z))
     
     #Fmax
     Fmaxf.append(z.max())
     #Fmin 
     Fminf.append(z.min())
     #print('Fmax =', Fmax, '\nFmin =', Fmin)
     
     #Return the sum of frequencies
     Fsumf.append(z.sum())
     #print('sum =', Fsum)
     #Average
     Favgf.append(z.mean())
     #print('Favg =', Favg)
     Frangef.append(Fmaxf[index] - Fminf[index])
     np.savez(outputFile, Vmax=Vmaxf, Vmin=Vminf, Vsum=Vsumf, Vavg=Vavgf, Vrange=Vrangef, Fmax=Fmaxf, Fmin=Fminf, Fsum=Fsumf, Favg=Favgf, Frange=Frangef)
     
p = 0
r = 0
i = 0
m = 0
t = 0
f = 0
n = 0

#load and feature extract data from a given path
def loadData(PATH, ):
    global p
    global r
    global i
    global m
    global t
    global f
    global n
    index = 0 # makes sure that data is not overwritten in pickle
    #Read the data for each day
    for file in os.listdir(PATH):
        newPATH = os.path.join(PATH, file)
        # Read the data for each dataset in each day's folder: fist, null, etc.
        for file in os.listdir(newPATH):
            loadFile = os.path.join(newPATH, file)
            data = pd.read_csv(loadFile, skiprows=7)
            data.columns = ['Sample', 'Date', 'CHANNEL1', 'Events']
            folderName = file.split('.')[0]
            PATH2 = 'C://Users/OwensLab/Documents/Preston/results/'
            outputFile = os.path.join(PATH2, folderName + '/outputData.npz')
            if (folderName == 'fist'):
                index = f
                fistExtraction(data, outputFile, index)
                f += 1
            elif (folderName == 'null'):
                index = n
                nullExtraction(data, outputFile, index)
                n += 1
            elif (folderName == 'thumb'):
                index = t
                thumbExtraction(data, outputFile, index)
                t+= 1
            elif (folderName == 'index'):
                index = i
                indexExtraction(data, outputFile, index)
                i+=1
            elif (folderName == 'middle'):
                index = m
                middleExtraction(data, outputFile, index)
                m+=1
            elif (folderName == 'ring'):
                index = r
                ringExtraction(data, outputFile, index)
                r+=1
            else:
                index = p
                pinkieExtraction(data, outputFile, index)
                p+=1
            
loadData(PATH)
outputFile.close()