# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:08:47 2020

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

clt = DBSCAN(eps = 0.28, min_samples = 14)
datalables = clt.fit_predict(data_zs)

r1 = pd.Series(datalables).value_counts()

print(r1)

datapro = np.column_stack((data ,datalables))

highdata = datapro[datapro[:,8] == 0]
lowdata = datapro[datapro[:,8] == -1]

np.savetxt('highdata.txt', highdata)
np.savetxt('lowdata.txt',  lowdata)

plt.figure(1)
plt.scatter(lowdata[:,3], lowdata[:,4], marker='o', color='grey',s=5.0)
plt.scatter(highdata[:,3], highdata[:,4], marker='o', color='lightcoral',s=5.0)
pmdata = np.vstack((highdata[:,3], highdata[:,4]))
np.savetxt('PM.txt', pmdata)
plt.xlabel('pmRA',fontsize=14)
plt.ylabel('pmDEC',fontsize=14)
plt.xlim((-15,15))
plt.ylim((-15,15))

plt.figure(10)
plt.hist(lowdata[:,4], bins=500, density = 1, facecolor='blue', alpha=0.5)
plt.hist(highdata[:,4], bins=50, density = 1, facecolor='red', alpha=0.5)



plt.figure(2)
hparallax = highdata[:,2]
hGmag = highdata[:,5]

lparallax = lowdata[:,2]
lGmag = lowdata[:,5]

plt.scatter(lGmag, lparallax, marker='o', color='grey',s=5)
plt.scatter(hGmag, hparallax, marker='o', color='lightcoral',s=5)
pallaxdata = np.vstack((lGmag, lparallax))
np.savetxt('parallax.txt', pallaxdata)
plt.xlabel('Gmag',fontsize=14)
plt.ylabel('parallax',fontsize=14)
plt.figure(21)
plt.hist(lowdata[:,2], bins=500, density = 1, facecolor='blue', alpha=0.5)
plt.hist(highdata[:,2], bins=50, density = 1, facecolor='red', alpha=0.5)


plt.figure(3)
highdataGmag = highdata[:,5]
highdataBPRP = highdata[:,6]-highdata[:,7]
loaddata = np.vstack((highdataGmag,highdataBPRP))
np.savetxt('BPRPG.txt', loaddata)
plt.xlim((-1,4))
plt.ylim((10,22))
plt.scatter((lowdata[:,6]-lowdata[:,7]), lowdata[:,5], marker='o', color='grey',s=5)
plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=5)
x_major_locator = MultipleLocator(1)
plt.xlabel('BP-RP',fontsize=14)
#plt.xlabel('G-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


plt.figure(5)

ax1 = plt.axes(projection='3d')
ax1.scatter3D(lowdata[:,0], lowdata[:,1], lowdata[:,2], c = 'b', marker='o', s=0.01)
ax1.scatter3D(highdata[:,0], highdata[:,1], highdata[:,2], c ='r', marker='o', s=1)
ax1.set_xlabel('RA')
#ax1.set_xlim(-6, 4)  #拉开坐标轴范围显示投影
ax1.set_ylabel('DEC')
#ax1.set_ylim(-4, 6)
ax1.set_zlabel('Parallax')
#ax1.set_zlim(-2, 2)
ax1.set_title('NGC7142')

#ax1.view_init(elev=30, azim=30)

plt.figure(11)
plt.hist(lowdata[:,0], bins=500, density = 1, facecolor='blue', alpha=0.5)
plt.hist(highdata[:,0], bins=50, density = 1, facecolor='red', alpha=0.5)

'''
plt.figure(6)
ax1 = plt.axes(projection='3d')

gif_images = []
for t in range (0,1000):
    if t == 360:
        break
    plt.cla()
    
    #ax1.set_zlim(-5, 5)
    ax1.scatter3D(lowdata[:,0], lowdata[:,1], lowdata[:,2], c = 'b', marker='o', s=0.01)
    ax1.scatter3D(highdata[:,0], highdata[:,1], highdata[:,2], c ='r', marker='o', s=1)
    
    ax1.set_xlabel('RA')
    ax1.set_ylabel('DEC')
    ax1.set_zlabel('Parallax')

    plt.pause(0.01)
    plt.savefig('1.jpg')
    
    ax1.view_init(elev=30, azim=t+1)
    gif_images.append(imageio.imread('1.jpg'))
    
imageio.mimsave("NGC7142.gif",gif_images,fps=20)
'''
    
'''
plt.figure(4)
dataable = np.column_stack((data_zs ,datalables))
pddata = pd.DataFrame(dataable)
datazs = pd.DataFrame(data_zs)

tsne = TSNE(n_components=2, learning_rate=100, n_iter=1000, init='pca')
tsne.fit_transform(data_zs)    #进行降维
tsne = pd.DataFrame(tsne.embedding_, index=datazs.index)    #转换数据格式

d = tsne[pddata.iloc[:,5] == -1]
plt.scatter(d[0], d[1], c = 'b', s = 0.1)

#d = tsne[pddata.iloc[:,5] == 0]
#plt.scatter(d[0], d[1], c = 'r', s = 5)

'''