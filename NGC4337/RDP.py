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

data = np.loadtxt('NGC4337.txt')
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

clt = DBSCAN(eps = 0.21, min_samples = 14)
datalables = clt.fit_predict(data_zs)

r1 = pd.Series(datalables).value_counts()

print(r1)

datapro = np.column_stack((data ,datalables))

highdata = datapro[datapro[:,8] == 0]
lowdata = datapro[datapro[:,8] == -1]

np.savetxt('highdata.txt', highdata)
np.savetxt('lowdata.txt',  lowdata)

temp = [0 for i in range (50)]
def lendata(datax,RAmean, DECmean):
    data = np.copy(datax)
    lendata = len(data)
    for i in range(lendata):
        x = data[i][0]
        y = data[i][1]
        d = np.sqrt((x-RAmean)**2+(y-DECmean)**2)
        
        for j in range(0,50):
            if (d<(j+1)/60 and d>j/60):
                temp[j] = temp[j]+1
               
    for i in range(0,50):
        s = ((i+1))**2*np.pi - (i)**2*np.pi   
        temp[i] = np.float64(temp[i])/s
    return temp

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
xr = np.arange(0,50,1)
#plt.plot(np.log10(xr), np.log10(temp), '.')
plt.plot(xr, temp, '.')