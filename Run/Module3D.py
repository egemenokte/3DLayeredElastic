"""
Created on Sat Jun  5 19:57:57 2021

@author: egeme

Multi dimensional analysis built on top of MLE. Experimental
"""

import sys
import os
sys.path.append('..')
from Main.MDA import Layer3D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%% POSITIVE IS COMPRESSION
L=[9,9] #Load Magnitudes (kip)
LPos=[(19,0),(31,0)] #load positions as (x,y) (inch)
a = 5                   # contact radius (inch)
x = np.arange(0,51,1)                 # x query points (inch) 
y=[0]                                 # y query points (inch) 
z = np.arange(0,20,1)                 # z (depth) query points (inch)
H = [8, 6]                 # Layer Thicknesses (inch), subgrade is not required and assumed infinite
E = [500, 40, 15]           # Layer Modulus (ksi)
nu = [0.35, 0.4, 0.45]        # Poissons ratio

## Do not change unless slow or unstable
ZRO=7*1e-22 
isBD=[1,1]
it = 1600                    # number of iterations
tolerance=0.001 #percent error

#%%
#Solve    
RS=Layer3D(L,LPos,a,x,y,z,H,E,nu,it,ZRO,isBD,tolerance)  

#%%
#Plots the 0th y axis   

plt.close('all')
response='eps_z' 
A=np.transpose(RS[response][0,:,:])*10**6
# A=RS['sigma_z'][0,:,:]
sns.heatmap(A,xticklabels=x,yticklabels=-z,cbar_kws={'label': '$\mu\epsilon$'}) #positive is compression
plt.xlabel('x')
plt.ylabel('z')
plt.title(response)

plt.figure()
response='eps_y' 
A=np.transpose(RS[response][0,:,:])
# A=RS['sigma_z'][0,:,:]
sns.heatmap(A,xticklabels=x,yticklabels=-z,cbar_kws={'label': '$\mu\epsilon$'})
plt.xlabel('x')
plt.ylabel('z')
plt.title(response)

plt.figure()
response='sigma_z' 
A=np.transpose(RS[response][0,:,:])
# A=RS['sigma_z'][0,:,:]
sns.heatmap(A,xticklabels=x,yticklabels=-z,cbar_kws={'label': 'Stress (psi)'})
plt.xlabel('x')
plt.ylabel('z')
plt.title(response)

fig, ax = plt.subplots(figsize=(10, 8))
response='deflection_z' 
mag=50
X,Z=np.meshgrid(x,z)
A=np.transpose(RS[response][0,:,:])*mag
# A=RS['sigma_z'][0,:,:]
ax.scatter(X.flatten(),-Z.flatten(),s=0.5)
ax.scatter(X.flatten(),-Z.flatten()-A.flatten())
ax.set_xticks(x)
ax.set_yticks(-z)
plt.title('Deformation field magnified '+str(mag)+' times')
plt.xlabel('x')
plt.ylabel('z')
# plt.grid()

ax.quiver(X,-Z,-A*0,-A,-A, scale_units='xy', scale=1.)
