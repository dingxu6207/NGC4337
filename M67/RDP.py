# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:09:42 2021

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

temp = [0 for i in range (120)]
def lendata(datax,RAmean, DECmean):
    data = np.copy(datax)
    lendata = len(data)
    for i in range(lendata):
        x = data[i][0]
        y = data[i][1]
        d = np.sqrt((x-RAmean)**2+(y-DECmean)**2)
        
        for j in np.arange(0,60,0.5):
            if (d<(j+0.5)/60 and d>=j/60):
                index = np.int(j*2)
                temp[index] = temp[index]+1
               
    for i in np.arange(0,60,0.5):
        s = ((i+0.5))**2*np.pi - (i)**2*np.pi  
        index = np.int(i*2)
        temp[index] = np.float64(temp[index])/s
    return temp


def func(x,fbg,f0, rc):
    return fbg+f0/(1+(x/rc)**2)

plt.figure(1)
plt.scatter(lowdata[:,0], lowdata[:,1], c = 'b', marker='o', s=0.01)
plt.scatter(highdata[:,0], highdata[:,1], c ='r', marker='o', s=1)
plt.xlabel('RA',fontsize=14)
plt.ylabel('DEC',fontsize=14)
plt.plot(np.mean(highdata[:,0]),np.mean(highdata[:,1]),'o',c='g')



plt.figure(2)
plt.hist(lowdata[:,0], bins=500, density = 1, facecolor='blue', alpha=0.5)
plt.hist(highdata[:,0], bins=10, density = 1, facecolor='red', alpha=0.5)

print('RAmean = ', np.mean(highdata[:,0]))
print('RAstd = ', np.std(highdata[:,0]))

print('DECmean = ', np.mean(highdata[:,1]))
print('DECstd = ', np.std(highdata[:,1]))

RAmean  = np.mean(highdata[:,0])
DECmean = np.mean(highdata[:,1])
temp = lendata(data,RAmean, DECmean)

plt.figure(3)
xr = np.arange(0.5,60.5,0.5)
#plt.plot(np.log10(xr), np.log10(temp), '.')
plt.plot(xr, temp, '.')

popt, pcov = curve_fit(func, xr, temp)


print( popt)
plt.plot(xr, func(xr, *popt), 'r-',
label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

alllist = []
listxr = xr.tolist()
alllist.append(listxr)
alllist.append(temp)

arraylist = np.array(alllist)
np.savetxt('arraylist.txt', arraylist)
