# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 19:57:57 2021

@author: egeme

Multi dimensional analysis built on top of MLE. Experimental
"""
import numpy as np
# from Main.MLEV3 import PyMastic
from Main.MLEV_Parallel import PyMastic
def Layer3D(L,LPos,a,x,y,z,H,E,nu,it,ZRO=7*1e-22 ,isBD=[1,1],tolerance=10**-6,verbose=True,every=100):
    #First, we are going to calculate x,y and z stresses from z,r and t. Then we are going to use superposition to add those up
    #At the end, with everything in place, we are going to calculate strains
    tolerance=tolerance/100
    it=int(it/4) # number of iterations
    # newkeys=['eps_x','eps_y','eps_z','sigma_x','sigma_y','sigma_z','sigma_xy','deflection_z']
    newkeys=['deflection_z','sigma_x','sigma_y','sigma_z','sigma_xy','sigma_yz','sigma_xz','eps_x','eps_y','eps_z','eps_xy','eps_yz','eps_xz']
    RS={}
    
    for key in newkeys: #Create the empty dictionary to store the values of stress
        RS[key]=np.zeros((len(y),len(x),len(z)))
    DRS={}
    
    xx,yy=np.meshgrid(x,y) #create a mesh grid such that each x and each y will be calculated together
    for i in range(len(L)): #Loop through each load that is applied
        if verbose:
            print('Load ', str(i+1),'/',str(len(L)))
            
        #We care about relative distances from the load application point, not the absolute x and y for the calculations
        xxt=xx-LPos[i][0]+1e-22 
        yyt=yy-LPos[i][1]
        
        t=np.arctan2(yyt,xxt) #calculate the angle between -180 and 180
        # cost=np.cos(np.arctan(yyt/(xxt+1e-12)))
        # sint=np.sin(np.arctan(yyt/(xxt+1e-12)))
        pts=np.sqrt(xxt**2+yyt**2) #these are our "r" values that will be calculated
        unique_pts=np.unique(pts) #Find the unique ones to save time
        #Calculate the stresses (and strains but we are not using them yet)
        try:
            DRS[i] = PyMastic(L[i]/np.pi/a**2,a,unique_pts,z,H,E,nu, ZRO, isBounded = np.ones(len(H)), iteration = it, inverser = 'solve',tol=tolerance,every=every,verbose=verbose)
        except:
            DRS[i] = PyMastic(L[i]/np.pi/a**2,a,unique_pts,z,H,E,nu, ZRO, isBounded = np.ones(len(H)), iteration = it, inverser = 'solve',tol=tolerance,every=20,verbose=verbose)
        # We have to convert the unique points back into the grid form
        for j in range(len(unique_pts)):
            query=np.where(pts==unique_pts[j]) #Find in the original grid, where they are
            for k in range(len(query[0])): #Loop through them and assign
                idx=[query[0][k],query[1][k]]
                                          
                # RS['eps_x'][idx[0],idx[1],:]=RS['eps_x'][idx[0],idx[1],:]+DRS[i]['Strain_R'][:,j]*np.cos(t[idx[0],idx[1]])+DRS[i]['Strain_T'][:,j]*np.abs(np.sin(t[idx[0],idx[1]]))
                # RS['eps_y'][idx[0],idx[1],:]=RS['eps_y'][idx[0],idx[1],:]+DRS[i]['Strain_R'][:,j]*np.sin(t[idx[0],idx[1]])+DRS[i]['Strain_T'][:,j]*np.abs(np.cos(t[idx[0],idx[1]]))
                #Huang book chapter 6 page 96
                RS['deflection_z'][idx[0],idx[1],:]=RS['deflection_z'][idx[0],idx[1],:]+DRS[i]['Displacement_Z'][:,j]
                RS['sigma_z'][idx[0],idx[1],:]=RS['sigma_z'][idx[0],idx[1],:]+DRS[i]['Stress_Z'][:,j]
                RS['sigma_x'][idx[0],idx[1],:]=RS['sigma_x'][idx[0],idx[1],:]+DRS[i]['Stress_R'][:,j]*np.cos(t[idx[0],idx[1]])**2+DRS[i]['Stress_T'][:,j]*np.sin(t[idx[0],idx[1]])**2
                RS['sigma_y'][idx[0],idx[1],:]=RS['sigma_y'][idx[0],idx[1],:]+DRS[i]['Stress_R'][:,j]*np.sin(t[idx[0],idx[1]])**2+DRS[i]['Stress_T'][:,j]*np.cos(t[idx[0],idx[1]])**2
                RS['sigma_xy'][idx[0],idx[1],:]=RS['sigma_xy'][idx[0],idx[1],:]+(DRS[i]['Stress_R'][:,j]-DRS[i]['Stress_T'][:,j])*np.sin(t[idx[0],idx[1]])*np.cos(t[idx[0],idx[1]])
                RS['sigma_yz'][idx[0],idx[1],:]=RS['sigma_yz'][idx[0],idx[1],:]+DRS[i]['Stress_RZ'][:,j]*np.sin(t[idx[0],idx[1]])
                RS['sigma_xz'][idx[0],idx[1],:]=RS['sigma_xz'][idx[0],idx[1],:]+DRS[i]['Stress_RZ'][:,j]*np.cos(t[idx[0],idx[1]])
                # RS['eps_z'][idx[0],idx[1],:]=RS['eps_z'][idx[0],idx[1],:]+DRS[i]['Strain_Z'][:,j]   
        
        RS=calculate_strain_2(RS,H,E,nu,z)
                
    return RS

def calculate_strain(RS,H,E,nu,z): #there is an error in this function, fixed with calculate_strain_2
    Eglobal=RS['sigma_z']*0
    nuglobal=RS['sigma_z']*0
    Hsum=np.cumsum(H)
    layer=0 #first layer
    for i in range(len(z)):
        if z[i]<Hsum[layer]:
            Eglobal[:,:,i]=E[layer]
            nuglobal[:,:,i]=nu[layer]
            continue
        
        layer=layer+1
        if layer==len(H):
            Eglobal[:,:,i:]=E[layer]
            nuglobal[:,:,i:]=nu[layer]
            break
        Eglobal[:,:,i]=E[layer]
        nuglobal[:,:,i]=nu[layer]
        
    RS['eps_z']=1/Eglobal*(RS['sigma_z']-nuglobal*(RS['sigma_x']+RS['sigma_y']))
    RS['eps_x']=1/Eglobal*(RS['sigma_x']-nuglobal*(RS['sigma_z']+RS['sigma_y']))
    RS['eps_y']=1/Eglobal*(RS['sigma_y']-nuglobal*(RS['sigma_z']+RS['sigma_x']))
    RS['eps_xy']=2*(1+nuglobal)/Eglobal*RS['sigma_xy']
    RS['eps_yz']=2*(1+nuglobal)/Eglobal*RS['sigma_yz']
    RS['eps_xz']=2*(1+nuglobal)/Eglobal*RS['sigma_xz']
    return RS

def calculate_strain_2(RS,H,E,nu,z): #Updated version of global E and nu calculation
    Eglobal=RS['sigma_z']*0
    nuglobal=RS['sigma_z']*0
    Hsum=np.copy(np.cumsum(H))
    Hsum=np.append(Hsum,10**6)
    for i in range(len(z)):
        layer=np.argmax(z[i]<Hsum)
        Eglobal[:,:,i]=E[layer]
        nuglobal[:,:,i]=nu[layer]
        
    RS['eps_z']=1/Eglobal*(RS['sigma_z']-nuglobal*(RS['sigma_x']+RS['sigma_y']))
    RS['eps_x']=1/Eglobal*(RS['sigma_x']-nuglobal*(RS['sigma_z']+RS['sigma_y']))
    RS['eps_y']=1/Eglobal*(RS['sigma_y']-nuglobal*(RS['sigma_z']+RS['sigma_x']))
    RS['eps_xy']=2*(1+nuglobal)/Eglobal*RS['sigma_xy']
    RS['eps_yz']=2*(1+nuglobal)/Eglobal*RS['sigma_yz']
    RS['eps_xz']=2*(1+nuglobal)/Eglobal*RS['sigma_xz']
    return RS

def NRhapson(L,LPos,a,x,y,z,H,Ei,nu,it,ZRO,isBD,tolerance,Expected,breaktol=0.1,ss=np.array([1,0.1,0]),delta=200,verbose=False): #Iterative algorithm
    RS=Layer3D(L,LPos,a,x,y,z,H,Ei,nu,it,ZRO,isBD,tolerance,verbose=False)
    TT=np.transpose(RS['deflection_z'][0,:,:]).flatten()
    Loss=np.mean(np.abs((TT-Expected)/Expected*100))
    Ls=[]
    for i in range(200):
        RS=Layer3D(L,LPos,a,x,y,z,H,Ei+np.array([ss[0],0,0]),nu,it,ZRO,isBD,tolerance,verbose=False)
        TT=np.transpose(RS['deflection_z'][0,:,:]).flatten()
        Loss2=np.mean(np.abs((TT-Expected)/Expected*100))
        d1=(Loss2-Loss)/ss[0]
        RS=Layer3D(L,LPos,a,x,y,z,H,Ei+np.array([0,ss[1],0]),nu,it,ZRO,isBD,tolerance,verbose=False)
        TT=np.transpose(RS['deflection_z'][0,:,:]).flatten()
        Loss2=np.mean(np.abs((TT-Expected)/Expected*100))
        d2=(Loss2-Loss)/ss[1]
        Ei=Ei-delta*np.array([d1*ss[0],d2*ss[1],0])
        RS=Layer3D(L,LPos,a,x,y,z,H,Ei,nu,it,ZRO,isBD,tolerance,verbose=False)
        TT=np.transpose(RS['deflection_z'][0,:,:]).flatten()
        Loss=np.mean(np.abs((TT-Expected)/Expected*100))
        
        Ls.append(Loss)
        print('Step', i+1,'Loss',np.round(Loss,5),'%')
        # print(Loss,d1,d2)
        if np.abs(Loss)<breaktol:
            print(Ei)
            return Ei,Ls,RS
    print('Reached Iteration Limit')
    return np.array([]),Ls,RS