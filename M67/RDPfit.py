# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:44:19 2021

@author: dingxu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:18:42 2020

@author: dingxu
"""

import numpy as np
import emcee
import matplotlib.pyplot as pl
from matplotlib.pyplot import cm 
import matplotlib.mlab as mlab
from scipy.stats import norm
import matplotlib.pyplot as plt

#def q(a,b,c,x):
#    quad = a*x**2. + b*x + c
#    return quad
#
#x = np.linspace(-1,1,101)
#
#quad = q(1.,0.,-0.2,x)
#noise = np.random.normal(0.0, 0.1, quad.shape)
#noisy = quad + noise
#plt.figure(0)
#pl.plot (x,noisy,"k.")
#pl.show()

def q(fbg,f0, rc, x):
    return fbg+f0/(1+(x/rc)**2)

data = np.loadtxt('arraylist.txt')
x = data[0,:]
noisy = data[1,:]


nwalkers = 30
niter = 250
init_dist = [(0.,2.),(4.5,6),(3.5,5)]
ndim = len(init_dist)
sigma = np.diff(noisy,2).std()/np.sqrt(6)
#sigma = 0.05

priors = [(0.,2.),(4.5,6.),(3.5,5.)]

def rpars(init_dist):
    return [np.random.rand() * (i[1]-i[0]) + i[0] for i in init_dist]


def lnprior(priors, values):
    
    lp = 0.
    for value, prior in zip(values, priors):
        if value >= prior[0] and value <= prior[1]:
            lp+=0
        else:
            lp+=-np.inf 
    return lp


def lnprob(z):
    
    lnp = lnprior(priors,z)
    if not np.isfinite(lnp):
            return -np.inf

    # make a model using the values the sampler generated
    model = q(z[0],z[1],z[2],x)

    # use chi^2 to compare the model to the data:
    chi2 = 0.
    for i in range (len(x)):
            chi2+=((noisy[i]-model[i])**2)/(sigma**2)

    # calculate lnp
    lnprob = -0.5*chi2 + lnp

    return lnprob


def run(init_dist, nwalkers, niter, ndim):

    # Generate initial guesses for all parameters for all chains
    p0 = np.array([rpars(init_dist) for i in range(nwalkers)])
    #print(p0)

    # Generate the emcee sampler. Here the inputs provided include the 
    # lnprob function. With this setup, the first parameter
    # in the lnprob function is the output from the sampler (the paramter 
    # positions).
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob)

    pos, prob, state = sampler.run_mcmc(p0, niter)

    for i in range(ndim):
        pl.figure(i+1)
        y = sampler.flatchain[:,i]
        n, bins, patches = pl.hist(y, 200, normed=1, color="b", alpha=0.45)
        pl.title("Dimension {0:d}".format(i))
        
        mu = np.average(y)
        sigma = np.std(y)       
        print ("mu,", "sigma = ", mu, sigma)

        bf = norm.pdf(bins, mu, sigma)
        l = pl.plot(bins, bf, 'k--', linewidth=2.0)

    pl.show()
    return pos

'''
niter=10
pos = run(init_dist, nwalkers, niter, ndim)


color=cm.rainbow(np.linspace(0,1,nwalkers))
for i,c in zip(range(nwalkers),color):
    
    model = pos[-1-i,0]*x**2 + pos[-1-i,1]*x + pos[-1-i,2]
    
   
    plt.figure(3)
    pl.plot(x,model,c=c)    
pl.plot(x,noisy,"k.")
pl.xlabel("x")
pl.ylabel("f(x)")
pl.show()


'''
niter = 10000

pos = run(init_dist, nwalkers, niter, ndim)


color=cm.rainbow(np.linspace(0,1,nwalkers))
for i,c in zip(range(nwalkers),color):
    
    model = pos[-1-i,0]+pos[-1-i,1]/(1+(x/pos[-1-i,2])**2)
    
    plt.figure(4)
    pl.plot(x,model,c=c)
    
pl.plot(x,noisy,"k.")
pl.xlabel("x")
pl.ylabel("f(x)")
pl.show()