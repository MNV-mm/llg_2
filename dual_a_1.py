#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:51:16 2023

@author: mnv
"""
# In[1] Матрицы поворота
import numpy as np

# Ku = -312
# Kp = 3338
# Kc = 1555

# tu = 50.2/180*np.pi
# psu = -203/180*np.pi
# psp = -185/180*np.pi

Ku = 852.9
Kp = 2440
Kc = 1275

tu = 40/180*np.pi
psu = 189.4/180*np.pi
psp = 9.44/180*np.pi

a1 = np.array([[ np.sin(psu),  np.cos(psu), 0.],
              [-np.cos(psu),  np.sin(psu), 0.],
              [0., 0., 1.]])

a2 = np.array([[ 1.,  0., 0.],
              [0., np.cos(tu), np.sin(tu)],
              [0., -np.sin(tu), np.cos(tu)]])

cos_g = -np.cos(tu)*np.cos(psu+psp)/np.sqrt(np.sin(psu+psp)**2 + np.cos(tu)**2*np.cos(psu+psp)**2)
sin_g = np.sin(psu+psp)/np.sqrt(np.sin(psu+psp)**2 + np.cos(tu)**2*np.cos(psu+psp)**2)

# cos_g = -np.cos(psu)/np.sqrt(np.sin(psu)**2*np.cos(tu)**2 + np.cos(psu)**2)
# sin_g = np.sin(psu)*np.cos(tu)/np.sqrt(np.sin(psu)**2*np.cos(tu)**2 + np.cos(psu)**2)

# a3 = np.array([[ 0.,  -1., 0.],
#               [1.,  0., 0.],
#               [0., 0., 1.]])

# a3 = np.array([[ sin_g,  cos_g, 0.],
#               [-cos_g,  sin_g, 0.],
#               [0., 0., 1.]])

a3 = np.array([[ np.cos(-0.29755104),  -np.sin(-0.29755104), 0.],
              [np.sin(-0.29755104),  np.cos(-0.29755104), 0.],
              [0., 0., 1.]])

a0 = np.array([[-2/np.sqrt(5), 0., 1/np.sqrt(5)],
               [1/np.sqrt(5), 0., 2/np.sqrt(5)],
               [0., 1., 0.]])

#a = a3.dot(a2.dot(a1))
#a = np.matmul(a3,np.matmul(a2,a1))
a = np.matmul(a2,a1)

#Nu = np.array([a[0,2], a[1,2], a[2,2]])
Nu = np.array([np.sin(tu)*np.cos(psu), np.sin(tu)*np.sin(psu), np.cos(tu)])
#Np_0 = np.array([np.cos(psp), np.sin(psp), 0.])
#Np = a.dot(Np_0)
#Np = np.matmul(a,Np_0)

sin_g = np.cos(tu)/np.sqrt(np.cos(tu)**2 + np.sin(tu)**2*np.cos(psu-psp)**2)
cos_g = -np.sin(tu)*np.cos(psu-psp)/np.sqrt(np.cos(tu)**2 + np.sin(tu)**2*np.cos(psu-psp)**2)
Np = np.array([sin_g*np.cos(psp), sin_g*np.sin(psp), cos_g])

#Mat = a.dot(a0)
#Mat = np.matmul(a,a0)
Mat_1 = np.transpose(a0)

def w(x, y, Nu, Np, Mat_1, Ku, Kp, Kc):
    m = np.array([np.sin(x)*np.cos(y), np.sin(x)*np.sin(y), np.cos(x)])
    #mm = Mat_1.dot(m)
    mm = np.matmul(Mat_1,m)
    #f = -Ku*(Nu.dot(m))**2 + Kp*(Np.dot(m))**2 + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2)
    f = -Ku*(np.matmul(Nu,m))*np.matmul(Nu,m) + Kp*(np.matmul(Np,m))*np.matmul(Np,m) + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2)
    return f

def w_s(x, Nu, Np, Mat_1, Ku, Kp, Kc):
    N = x.size//2
    #m = np.array([np.sin(x)*np.cos(y), np.sin(x)*np.sin(y), np.cos(x)])
    mx = np.sin(x[:N])*np.cos(x[N:])
    my = np.sin(x[:N])*np.sin(x[N:])
    mz = np.cos(x[:N])
    m_massive = [mx, my, mz]
    #mm = Mat_1.dot(m)
    mm = np.matmul(Mat_1,m_massive)
    #f = -Ku*(Nu.dot(m))**2 + Kp*(Np.dot(m))**2 + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2)
    #f = -Ku*(np.matmul(Nu,m))*np.matmul(Nu,m) + Kp*(np.matmul(Np,m))*np.matmul(Np,m) + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2)
    f = sum(-Ku*(Nu[0]*mx + Nu[1]*my + Nu[2]*mz)**2  + Kp*(Np[0]*mx + Np[1]*my + Np[2]*mz)**2 + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2))
    return f

# In[2] Минимизация энергии анизотропии
nx = 100
ny = 200

theta = np.linspace(0, np.pi, nx)
phi = np.linspace(0, 2*np.pi, ny)

test = np.zeros((nx,ny))

X = np.zeros((nx,ny))
Y = np.zeros((nx,ny))
Z = np.zeros((nx,ny))

for i in range(0, nx, 1):
    for j in range(0, ny, 1):
        test[i,j] = w(theta[i], phi[j], Nu, Np, Mat_1, Ku, Kp, Kc)
        
minimum = np.min(test)
test_n = test + abs(minimum)
max_n = np.max(test_n)
for i in range(0, nx, 1):
    for j in range(0, ny, 1):
        X[i,j] = test_n[i,j]/max_n*np.sin(theta[i])*np.cos(phi[j])
        Y[i,j] = test_n[i,j]/max_n*np.sin(theta[i])*np.sin(phi[j])
        Z[i,j] = test_n[i,j]/max_n*np.cos(theta[i])

def wm(x, Nu, Np, Mat_1, Ku, Kp, Kc):
    m = np.array([np.sin(x[0])*np.cos(x[1]), np.sin(x[0])*np.sin(x[1]), np.cos(x[0])])
    #mm = Mat_1.dot(m)
    mm = np.matmul(Mat_1,m)
    #f = -Ku*(Nu.dot(m))**2 + Kp*(Np.dot(m))**2 + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2)
    f = -Ku*(np.matmul(Nu,m))*np.matmul(Nu,m) + Kp*(np.matmul(Np,m))*np.matmul(Np,m) + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2)
    return f
bounds = [(0*np.pi/2, np.pi), (0*np.pi, 2*np.pi/2)]
from scipy import optimize
results = optimize.shgo(wm, bounds, args=(Nu, Np, Mat_1, Ku, Kp, Kc))
th_0, ph_0 = results.x

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
cont = ax.contourf(phi, theta, test, levels = 25)
ax.plot(results.x[1],results.x[0], marker ='o', color = 'Red')
ax.set_xlabel('$\phi$')
ax.set_ylabel('$\Theta$')
plt.colorbar(cont)
plt.show()

# In[3] Минимизация Интеграла

n = 51
L = 20
x = np.linspace(-L/2,L/2,n)
dx = L/(n-1)

bounds = []
for i in range((n-4)):
    if i == (n-3)/2:
        bounds.append((0, np.pi))
    else:
        bounds.append((0, np.pi))

for i in range((n-4)):
    bounds.append((0, 2*np.pi))

def int_w(x, dx, Nu, Np, Mat_1, Ku, Kp, Kc):
    N_h = x.size//2
    mx = np.sin(x[:N_h])*np.cos(x[N_h:])
    my = np.sin(x[:N_h])*np.sin(x[N_h:])
    mz = np.cos(x[:N_h])
    
    m_massive = [mx, my, mz]
    
    mm = np.matmul(Mat_1,m_massive)
    return ((x[0]-th_0)/dx)**2 + ((np.pi-th_0-x[-1])/dx)**2  + sum(((x[1:N_h]-x[N_h:-1])/dx)**2) + np.sin(th_0)**2*((x[N_h+1]-ph_0)/dx)**2 + np.sin(np.pi-th_0)**2*((x[-1]-(-ph_0+np.pi))/dx)**2\
        + sum(np.sin(x[1:N_h])**2*((x[N_h+1:]-x[N_h:-1])/dx)**2) +  sum(-Ku*(Nu[0]*mx + Nu[1]*my + Nu[2]*mz)**2  + Kp*(Np[0]*mx + Np[1]*my + Np[2]*mz)**2 + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2))/(Ku)

from scipy.optimize import dual_annealing, differential_evolution, shgo, basinhopping
#res = dual_annealing(int_w, bounds, args = (dx, Nu, Np, Mat_1, Ku, Kp, Kc)) #, maxiter = 3000, initial_temp = 10000
#res = differential_evolution(int_w, bounds, args = (dx, Nu, Np, Mat_1, Ku, Kp, Kc))
#res = shgo(int_w, bounds, args = (dx, Nu, Np, Mat_1, Ku, Kp, Kc))
x0 = np.zeros(2*(n-4))+np.pi/4
minimizer_kwargs = {"method":"L-BFGS-B", "args": (dx, Nu, Np, Mat_1, Ku, Kp, Kc), "bounds": bounds}
res = basinhopping(int_w, x0, niter=200, minimizer_kwargs=minimizer_kwargs)
t_array = res.x[:res.x.size//2]
p_array = res.x[res.x.size//2:]
