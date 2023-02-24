"""
Created on Sat Jun  5 19:57:57 2021
last modified 02/24/2023

@author: egemen okte

Multi dimensional analysis built on top of MLE. Experimental. Units have to be consistent, same with WinJulea
"""

import sys
import os
sys.path.append('..')
from Main.MDA_Huang import Layer3D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%% POSITIVE IS COMPRESSION

#Example 1, imperial units
L=[4500,4500] #Load Magnitudes (kip)
LPos=[(19,0),(19+11.5,0)] #load positions as (x,y) (inch)
a = 5                   # contact radius (inch)
x = np.arange(0,51,1)                 # x query points (inch) 
y=[0]  # y query points (inch) 
# y = np.arange(-5,6,1)                               
z = np.arange(0,11,1)     # z (depth) query points (inch)         
H=[8] # Layer Thicknesses (inch), subgrade is not required and assumed infinite        
E=[150000,15000] # Layer Modulus (ksi)
nu = [0.35, 0.4, 0.45]        # Poissons ratio
nu=[0.5,0.5]


#Example 2, SI units
L=[21676] #Load Magnitudes (kN)
LPos=[(0,0)] #load positions as (x,y) (mm)
a = 100                   # contact radius (mm)
x = np.arange(-200,200,10)                 # x query points (mm) 
y=[0]  # y query points (mm) 
                              
z = np.arange(0,500,10)     # z (depth) query points (mm) 
# z=[24.99]          
H = [25,37.5,62.5,150]                 # Layer Thicknesses (mm), subgrade is not required and assumed infinite
E = [3300,2000,1250,220,70]           # Layer Modulus (MPa)
nu = [0.35, 0.35, 0.35,0.3,0.3]        # Poissons ratio


## Do not change unless slow or unstable
ZRO=7*1e-22 
isBD=[1,1]
it = 1600                  # number of iterations
tolerance=0.01 #percent error

#%%
#Solve    
RS=Layer3D(L,LPos,a,x,y,z,H,E,nu,it,ZRO,isBD,tolerance)  

#%%
#Plots the 0th y axis   
z=np.array(z)
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

#%%
plt.figure()
response='sigma_z' 
A=np.transpose(RS[response][0,:,:])
# A=RS['sigma_z'][0,:,:]
sns.scatterplot(A[:,4],-z)
plt.xlabel('x')
plt.ylabel('z')
plt.title(response)