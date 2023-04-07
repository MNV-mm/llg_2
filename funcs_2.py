#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 20:10:13 2022

@author: mnv
"""
import fenics as fen
import sympy as sp
import numpy as np
def n_pair(Ly1,l1,Z1,Z01,n):
    y, z = sp.symbols('x[1] z')
    ly, z1 = sp.symbols('ly z1')
    Ly,l,Z,Z0 = sp.symbols('Ly l Z Z0')
    z_m = z-z1
    z_p = z+z1
    y_m = y-ly
    y_p = y+ly
    
    phi_1 = 2*z_m*(sp.atan(y_m/z_m) - sp.atan(y_p/z_m)) \
        + 2*z_p*(sp.atan(y_p/z_p) - sp.atan(y_m/z_p)) \
             - 2*y_m*sp.atanh(2*z*z1/(y_m**2+z**2+z1**2)) \
                 + 2*y_p*sp.atanh(2*z*z1/(y_p**2+z**2+z1**2)) ## y, z, ly, z1
    #phi_1 = sp.Heaviside(y+ly) - sp.Heaviside(y-ly) #sp.exp(-y**2)
    
    dy = l/2+Ly/4
    phi = -phi_1.subs([(y,y-dy), (ly,(l-Ly/2)/2), (z1,Z)]) + phi_1.subs([(y,y+dy), (ly,(l-Ly/2)/2), (z1,Z)])
    ## y,z,l,Ly,Z
    dy = 3*l/2
    i = 1
    while i < n:
        phi = phi + (-1)**i*(phi_1.subs([(y,y-dy), (ly,l/2), (z1,Z)]) - phi_1.subs([(y,y+dy), (ly,l/2), (z1,Z)]))
        dy = dy+l
        i += 1
    
    #phi_n = phi.subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)])
    #print(phi_n)
    #sp.plot(phi_n,(y,-2*n*l1,2*n*l1))
    
    hy = -sp.diff(phi,y)#.subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)])
    hz = -sp.diff(phi,z)#.subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)])
    #(y,-2*n*l1,2*n*l1)
    p1 = sp.plot(hy.subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)]),(y,-Ly1/2,Ly1/2), show = False)
    p2 = sp.plot(hz.subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)]),(y,-Ly1/2,Ly1/2), show = False)
    p1.append(p2[0])
    p1.show()
    
    hy_c = sp.ccode(hy)
    hz_c = sp.ccode(hz)
    # print(hy_c)
    # llog = sp.ln(y)
    # llog = sp.printing.ccode(llog)
    
    out = fen.Expression(('0',hy_c,hz_c), degree = 3, l = l1, Ly = Ly1, Z = Z1, z = Z01)#, degree = 3, z=Z-1
    return out

def w(x, y, Nu, Np, Mat_1, Ku, Kp, Kc):
    m = np.array([np.sin(x)*np.cos(y), np.sin(x)*np.sin(y), np.cos(x)])
    #mm = Mat_1.dot(m)
    mm = np.matmul(Mat_1,m)
    #f = -Ku*(Nu.dot(m))**2 + Kp*(Np.dot(m))**2 + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2)
    f = -Ku*(np.matmul(Nu,m))*np.matmul(Nu,m) + Kp*(np.matmul(Np,m))*np.matmul(Np,m) + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2)
    return f

def wm(x, Nu, Np, Mat_1, Ku, Kp, Kc):
    m = np.array([np.sin(x[0])*np.cos(x[1]), np.sin(x[0])*np.sin(x[1]), np.cos(x[0])])
    #mm = Mat_1.dot(m)
    mm = np.matmul(Mat_1,m)
    #f = -Ku*(Nu.dot(m))**2 + Kp*(Np.dot(m))**2 + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2)
    f = -Ku*(np.matmul(Nu,m))*np.matmul(Nu,m) + Kp*(np.matmul(Np,m))*np.matmul(Np,m) + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2)
    return f

def wm_p(x, Nu, Np, Mat_1, Ku, Kp, Kc):
    m = np.array([np.sin(np.pi/2)*np.cos(x[0]), np.sin(np.pi/2)*np.sin(x[0]), 0.])
    #mm = Mat_1.dot(m)
    mm = np.matmul(Mat_1,m)
    #f = -Ku*(Nu.dot(m))**2 + Kp*(Np.dot(m))**2 + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2)
    f = -Ku*(np.matmul(Nu,m))*np.matmul(Nu,m) + Kp*(np.matmul(Np,m))*np.matmul(Np,m) + Kc*(mm[0]**2*mm[1]**2 + mm[0]**2*mm[2]**2 + mm[1]**2*mm[2]**2)
    return f

def phi_angle(Ku, Kp, Kc, tu, psu, psp, phi_a = 0):
    a1 = np.array([[ np.sin(psu),  np.cos(psu), 0.],
                  [-np.cos(psu),  np.sin(psu), 0.],
                  [0., 0., 1.]])

    a2 = np.array([[ 1.,  0., 0.],
                  [0., np.cos(tu), np.sin(tu)],
                  [0., -np.sin(tu), np.cos(tu)]])
    
    a3 = np.array([[ np.cos(phi_a),  np.sin(phi_a), 0.],
                  [-np.sin(phi_a),  np.cos(phi_a), 0.],
                  [0., 0., 1.]])
    
    a0 = np.array([[-2/np.sqrt(5), 0., 1/np.sqrt(5)],
                   [1/np.sqrt(5), 0., 2/np.sqrt(5)],
                   [0., 1., 0.]])
    
    #a = np.matmul(a2,a1)
    a = np.matmul(a3,np.matmul(a2,a1))
    
    Nu = np.array([a[0,2], a[1,2], a[2,2]])
    Np_0 = np.array([np.cos(psp), np.sin(psp), 0.])
    Np = np.matmul(a,Np_0)
    
    Mat = np.matmul(a,a0)
    Mat_1 = np.transpose(Mat)
    
    nx = 100
    ny = 200
    
    theta = np.linspace(0, np.pi, nx)
    phi = np.linspace(0, 2*np.pi, ny)
    test = np.zeros((nx,ny))
    
    for i in range(0, nx, 1):
        for j in range(0, ny, 1):
            test[i,j] = w(theta[i], phi[j], Nu, Np, Mat_1, Ku, Kp, Kc)
    
    bounds = [(0*np.pi/2+1E-5, np.pi-1E-5), (0*np.pi+1E-5, 2*np.pi/2-1E-5)]
    from scipy import optimize
    results = optimize.shgo(wm, bounds, args=(Nu, Np, Mat_1, Ku, Kp, Kc))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cont = ax.contourf(phi, theta, test, levels = 25)
    ax.plot(results.x[1],results.x[0], marker ='o', color = 'Red')
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$\Theta$')
    plt.colorbar(cont)
    plt.show()
    
    test_p = np.zeros(ny)

    for j in range(0, ny, 1):
        test_p[j] = w(np.pi/2, phi[j], Nu, Np, Mat_1, Ku, Kp, Kc)
    
    bounds_p = [(0, 2*np.pi/2)]
    from scipy import optimize
    results_p = optimize.shgo(wm_p, bounds_p, args=(Nu, Np, Mat_1, Ku, Kp, Kc))

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.plot(phi,test_p)
    ax.plot(results_p.x[0], w(np.pi/2, results_p.x[0], Nu, Np, Mat_1, Ku, Kp, Kc), marker ='o', color = 'Red')
    plt.show()
    return results_p

def aa_min(Nu, Np, Mat_1, Ku, Kp, Kc):
    nx = 100
    ny = 200
    
    theta = np.linspace(0, np.pi, nx)
    phi = np.linspace(0, 2*np.pi, ny)
    test = np.zeros((nx,ny))
    
    for i in range(0, nx, 1):
        for j in range(0, ny, 1):
            test[i,j] = w(theta[i], phi[j], Nu, Np, Mat_1, Ku, Kp, Kc)
    
    bounds = [(np.pi/2, np.pi), (0*np.pi, 2*np.pi)]
    from scipy import optimize
    results = optimize.dual_annealing(wm, bounds, args=(Nu, Np, Mat_1, Ku, Kp, Kc))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cont = ax.contourf(phi, theta, test, levels = 25)
    ax.plot(results.x[1],results.x[0], marker ='o', color = 'Red')
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$\Theta$')
    plt.colorbar(cont)
    plt.show()
    #fig.savefig('s30_min.png', dpi = 300)
    return results