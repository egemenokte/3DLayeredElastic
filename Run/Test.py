import sys
import os
sys.path.append('..')
from Main.MLE import PyMastic
import numpy as np
import seaborn as sns

L=9 #kips
q = 1000.0                   # psi
a = 5.99                    # inch
a=6.5
x = np.arange(0,20,0.2)                 # number of columns in response
z = np.arange(0,15,0.5)       # number of rows in response
# x=[0]
# z=[7.99]
H = [8]                 # inch
E = [150, 15]           # ksi
nu = [0.35, 0.4]
ZRO = 7*1e-12                # to avoid numerical instability
isBD= [1]
it = 500

RS = PyMastic(L*1000/np.pi/a**2,a,x,z,H,E,nu, ZRO, isBounded = isBD, iteration = it, inverser = 'solve')
# RS = PyMastic(q,a,x,z,H,E,nu, ZRO, isBounded = isBD, iteration = it, inverser = 'solve')


# print("\nDisplacement [0, 0]: ")
# print(RS['Displacement_Z'][0, 0])

# print("\nSigma Z is[0, 0]: ")
# print(RS['Stress_Z'][0, 0])

# print("\nDisplacement_H is [0, 0]: ")
# print(RS['Displacement_H'][0, 0])

# print("\nSigma T is [0, 0]: ")
# print(RS['Stress_T'][0, 0])

# print("\nDisplacement [1, 1]: ")
# print(RS['Displacement_Z'][1, 0])

# print("\nSigma Z is [1, 1]: ")
# print(RS['Stress_Z'][1, 0])

# print("\nSigma R is [1, 1]: ")
# print(RS['Stress_R'][1, 0])

# print("\nSigma T is [1, 1]: ")
# print(RS['Stress_T'][1, 0])

plt.close('all')
A=RS['Stress_Z']
B=np.concatenate((np.fliplr(A),A),1)
sns.heatmap(B)