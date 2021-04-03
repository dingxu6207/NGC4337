# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:02:11 2021

@author: dingxu
"""

import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import imageio
from skimage import draw
from scipy.optimize import curve_fit

data = np.loadtxt('M67.txt')
print(len(data))
#data = data[data[:,2]>0]
#data = data[data[:,2]<1]

data = data[data[:,3]<15]
data = data[data[:,3]>-15]

data = data[data[:,4]<15]
data = data[data[:,4]>-15]


X = np.copy(data[:,0:5])


X = StandardScaler().fit_transform(X)
#X = MinMaxScaler().fit_transform(X)
data_zs = np.copy(X)

clt = DBSCAN(eps = 0.22, min_samples = 14)
datalables = clt.fit_predict(data_zs)

r1 = pd.Series(datalables).value_counts()

print(r1)

datapro = np.column_stack((data ,datalables))

highdata = datapro[datapro[:,8] == 0]
lowdata = datapro[datapro[:,8] == -1]

np.savetxt('highdata.txt', highdata)
np.savetxt('lowdata.txt',  lowdata)

arcmin = 60


def cdfdata(datamass,RAmean,DECmean):
    tempcdf = [0 for i in range (arcmin*2)]
    temp = [0 for i in range (arcmin*2)]
    datam = np.copy(datamass)
    lendata = len(datam)
    print(lendata)
    for i in range(lendata):
        x = datam[i][0]
        y = datam[i][1]
        d = np.sqrt((x-RAmean)**2+(y-DECmean)**2)
        
        for j in np.arange(0,arcmin,0.5):
            if (d<(j+0.5)/60):
                index = np.int(j*2)
                temp[index] = temp[index]+1
                
                
    for i in np.arange(0,arcmin,0.5):
        index = np.int(i*2)
        tempcdf[index] = np.float64(temp[index])/lendata
    
    return tempcdf
                
            
        
       
#highmass <=14.5  mediummass >14.5 and <=17     low-mass>17
    
higmass = highdata[highdata[:,5] <= 14.5]
lowmass = highdata[highdata[:,5] > 17]
medimmass = highdata[highdata[:,5] > 14.5]
medimmass = medimmass[medimmass[:,5] <= 17]

RAmean  = np.mean(highdata[:,0])
DECmean = np.mean(highdata[:,1])
tempmmass = cdfdata(medimmass, RAmean, DECmean)
templmass = cdfdata(lowmass, RAmean, DECmean)
temphmass = cdfdata(higmass, RAmean, DECmean)

plt.figure(3)
xr = np.arange(0.5,arcmin+0.5,0.5)
#plt.plot(np.log10(xr), np.log10(temp), '.')
A, = plt.plot(xr, temphmass)
B, = plt.plot(xr, templmass)
C, = plt.plot(xr, tempmmass)

plt.xlabel('R(arcmin)',fontsize=14)
plt.ylabel('F',fontsize=14)
plt.legend(handles=[A,B,C],labels=["high-mass","low-mass","medium-mass"],loc='upper left')


