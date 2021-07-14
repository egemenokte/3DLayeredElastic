# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 19:57:57 2021

@author: egeme

Multi dimensional analysis built on top of MLE. Experimental
"""
import numpy as np
from Main.MLEV2 import PyMastic
def Layer3D(L,LPos,a,x,y,z,H,E,nu,it,ZRO=7*1e-22 ,isBD=[1,1],tolerance=10**-6,verbose=True):

    tolerance=tolerance/100
    it=int(it/4) # number of iterations
    newkeys=['eps_x','eps_y','eps_z','sigma_x','sigma_y','sigma_z','deflection_z']
    RS={}
    for key in newkeys:
        RS[key]=np.zeros((len(y),len(x),len(z)))
    DRS={}
    xx,yy=np.meshgrid(x,y)
    for i in range(len(L)):
        if verbose:
            print('Load ', str(i+1),'/',str(len(L)))
        xxt=xx-LPos[i][0]+1e-22
        yyt=yy-LPos[i][1]
        # t=np.rad2deg(np.arctan(yyt/(xxt+1e-12)))
        t=np.arctan2(yyt,xxt)
        t=np.arctan(yyt/xxt)
        # cost=np.cos(np.arctan(yyt/(xxt+1e-12)))
        # sint=np.sin(np.arctan(yyt/(xxt+1e-12)))
        pts=np.sqrt(xxt**2+yyt**2)
        unique_pts=np.unique(pts)
        DRS[i] = PyMastic(L[i]*1000/np.pi/a**2,a,unique_pts,z,H,E,nu, ZRO, isBounded = isBD, iteration = it, inverser = 'solve',tol=tolerance)
        for j in range(len(unique_pts)):
            query=np.where(pts==unique_pts[j])
            for k in range(len(query[0])):
                idx=[query[0][k],query[1][k]]
                RS['eps_z'][idx[0],idx[1],:]=RS['eps_z'][idx[0],idx[1],:]+DRS[i]['Strain_Z'][:,j]
                RS['sigma_z'][idx[0],idx[1],:]=RS['sigma_z'][idx[0],idx[1],:]+DRS[i]['Stress_Z'][:,j]
                RS['deflection_z'][idx[0],idx[1],:]=RS['deflection_z'][idx[0],idx[1],:]+DRS[i]['Displacement_Z'][:,j]
                
                RS['eps_x'][idx[0],idx[1],:]=RS['eps_x'][idx[0],idx[1],:]+DRS[i]['Strain_R'][:,j]*np.cos(t[idx[0],idx[1]])+DRS[i]['Strain_T'][:,j]*np.abs(np.sin(t[idx[0],idx[1]]))
                RS['eps_y'][idx[0],idx[1],:]=RS['eps_y'][idx[0],idx[1],:]+DRS[i]['Strain_R'][:,j]*np.sin(t[idx[0],idx[1]])+DRS[i]['Strain_T'][:,j]*np.abs(np.cos(t[idx[0],idx[1]]))
                RS['sigma_x'][idx[0],idx[1],:]=RS['sigma_x'][idx[0],idx[1],:]+DRS[i]['Stress_R'][:,j]*np.cos(t[idx[0],idx[1]])+DRS[i]['Stress_T'][:,j]*np.abs(np.sin(t[idx[0],idx[1]]))
                RS['sigma_y'][idx[0],idx[1],:]=RS['sigma_y'][idx[0],idx[1],:]+DRS[i]['Stress_R'][:,j]*np.sin(t[idx[0],idx[1]])+DRS[i]['Stress_T'][:,j]*np.abs(np.cos(t[idx[0],idx[1]]))

    return RS