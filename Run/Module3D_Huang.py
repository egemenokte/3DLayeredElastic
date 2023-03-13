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


# #Example 2, SI units
# L=[21676] #Load Magnitudes (kN)
# LPos=[(0,0)] #load positions as (x,y) (mm)
# a = 100                   # contact radius (mm)
# x = np.arange(-200,200,10)                 # x query points (mm) 
# y=[0]  # y query points (mm) 
                              
# z = np.arange(0,500,10)     # z (depth) query points (mm) 
# # z=[24.99]          
# H = [25,37.5,62.5,150]                 # Layer Thicknesses (mm), subgrade is not required and assumed infinite
# E = [3300,2000,1250,220,70]           # Layer Modulus (MPa)
# nu = [0.35, 0.35, 0.35,0.3,0.3]        # Poissons ratio

# E = np.array([500, 50, 20])*1000           # Layer Modulus (psi), [Top Layer, Second to top layer,...,Subgrade]
# H = [6, 18]                 # Layer Thicknesses (inch), [Top Layer, Second to top layer,...,nth layer]. Subgrade is not required and assumed semi-infinite
# nu = [0.35, 0.4, 0.45]      # Poissons ratio, [Top Layer, Second to top layer,...,Subgrade]
# L=[9000]                       # Load Magnitudes (lbs)
# LPos=[(10,0)]               # Load positions as (x,y) (inch)
# a = 4                       # Contact radius (inch)
# x = np.arange(0,21,1)       # x query points for FWD (inch) 
# y=[0,10]                    # y query points (inch) 
# z =np.arange(0,30,1)        # z (depth) query points (inch)

#Example 3
E = np.array([500, 50, 10])*1000   # Layer Modulus (psi), [Top Layer, Second to top layer,...,Subgrade]
H = [6, 18]                        # Layer Thicknesses (inch), [Top Layer, Second to top layer,...,nth layer]. Subgrade is not required and assumed semi-infinite
nu = [0.35, 0.4, 0.45]             # Poissons ratio, [Top Layer, Second to top layer,...,Subgrade]
L=[9000,9000]                      # Load Magnitudes (lbs)
LPos=[(10,0),(20,0)]               # Load positions as (x,y) (inch)
a = 4                              # Contact radius (inch)
x = np.arange(0,30,1)              # x query points for (inch) 
y=[0]                              # y query points (inch) 
z =np.arange(0,30,1)               # z (depth) query points (inch)

## Do not change unless slow or unstable
ZRO=7*1e-22 
isBD=[1,1]
isBD=np.ones(len(E))                # number of iterations
tolerance=0.001 #percent error


############################ DO NOT CHANGE BELOW THIS LINE#####################
## Layered elastic analysis settings. Do not change unless slow or unstable
ZRO=7*1e-20
it = 1600            # number of maximum iterations
tolerance=0.05     #average percent error of query points
every=100 #check for convergence every x steps
sns.set(rc={'figure.figsize':(20,10)},font_scale=1.15)

#%%
#Solve    
RS=Layer3D(L,LPos,a,x,y,z,H,E,nu,it,ZRO,isBD,tolerance,verbose=True,every=every)


#%% for interactive plots
# import plotly.express as px
# import plotly.io as io
# io.renderers.default='browser'
# z=np.array(z)
# plt.close('all')
# response='eps_z' 
# A=np.transpose(RS[response][0,:,:])
# fig = px.imshow(A)
# fig.show()
#%%
#Plots the 0th y axis   
z=np.array(z)
plt.close('all')
response='eps_z' 
A=np.transpose(RS[response][0,:,:])*10**6
# A=RS['sigma_z'][0,:,:]
heatmap=sns.heatmap(A,xticklabels=x,yticklabels=-z,cbar_kws={'label': '$\mu\epsilon$'}) #positive is compression
plt.xlabel('x')
plt.ylabel('z')
plt.title(response)

plt.figure()
response='eps_y' 
A=np.transpose(RS[response][0,:,:])*10**6
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
sns.scatterplot(x=A[:,9],y=-z)
plt.xlabel('x')
plt.ylabel('z')
plt.title(response)
#%%
plt.figure()
response='sigma_z' 
A=np.transpose(RS[response][0,:,:])
# A=RS['sigma_z'][0,:,:]
sns.scatterplot(x=x,y=A[9,:])
plt.xlabel('x')
plt.ylabel('z')
plt.title(response)