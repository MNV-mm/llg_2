#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 22:53:41 2023

@author: mnv
"""
# In[part 0]

from fenics import *
# import DD_Hd
# import funcs_2
import numpy as np
import math
import os
import sympy as sp
from sympy.printing import print_ccode
from ufl.operators import diff
from ufl import variable
#import bempp.api
def norm_sol(m, u, FS):
    #vector().array() is replaced by vector().get_local()
    u_array = u.vector().get_local()
    m_array = m.vector().get_local()
    N = int(np.size(u_array))
    v2d = vertex_to_dof_map(FS)
    d2v = dof_to_vertex_map(FS)
    u_array_2 = u_array[v2d]
    m_array_2 = m_array[v2d]
    mm_array_2 = m_array_2+u_array_2
    i = 0
    while i+2 < N:
        norm = math.sqrt(math.pow(mm_array_2[i],2) + math.pow(mm_array_2[i+1],2) + math.pow(mm_array_2[i+2],2))
        mm_array_2[i] /= norm
        mm_array_2[i+1] /= norm
        mm_array_2[i+2] /= norm
        i += 4
    
    mm_array = mm_array_2[d2v]
    m.vector()[:] = mm_array
    return m

def norm_sol_s(u, FS):
    #vector().array() is replaced by vector().get_local()
    u_array = u.vector().get_local()
    N = int(np.size(u_array))
    v2d = vertex_to_dof_map(FS)
    d2v = dof_to_vertex_map(FS)
    u_array_2 = u_array[v2d]
    i = 0
    while i+2 < N:
        norm = math.sqrt(math.pow(u_array_2[i],2) + math.pow(u_array_2[i+1],2) + math.pow(u_array_2[i+2],2))
        u_array_2[i] /= norm
        u_array_2[i+1] /= norm
        u_array_2[i+2] /= norm
        i += 3
    
    u_array = u_array_2[d2v]
    u.vector()[:] = u_array
    return u

def max_norm(u):
    FS = u.function_space()
    u_array = u.vector().get_local()
    N = int(np.size(u_array))
    v2d = vertex_to_dof_map(FS)
    d2v = dof_to_vertex_map(FS)
    u_array_2 = u_array[v2d]
    i = 0
    norm_prev = 0
    while i+2 < N:
        norm = math.sqrt(math.pow(u_array_2[i],2) + math.pow(u_array_2[i+1],2) + math.pow(u_array_2[i+2],2))
        if norm > norm_prev:
            norm_prev = norm
        i += 3
    return norm_prev

def h_rest(m, kku, kkp, kkc, M_s, nu, np, at):
    m1, m2, m3 = split(m)
    # e1, e2, e3 = split(e_f)
    # dedz_1, dedz_2, dedz_3 = split(dedz)
    ## есть маг поле по y
    # vec = as_vector((p*(2*e1*m1.dx(0) + 2*e2*m2.dx(0) + 2*e3*m3.dx(0) + m1*e1.dx(0) + m2*e2.dx(0) + m3*e3.dx(0) + m1*e1.dx(0) + m2*e1.dx(1) + m3*dedz_1), \
    #                  p*(2*e1*m1.dx(1) + 2*e2*m2.dx(1) + 2*e3*m3.dx(1) + m1*e1.dx(1) + m2*e2.dx(1) + m3*e3.dx(1) + m1*e2.dx(0) + m2*e2.dx(1) + m3*dedz_2), \
    #                       p*(m1*e3.dx(0) + m2*e3.dx(1) + m3*dedz_3 + m1*dedz_1 + m2*dedz_2 + m3*dedz_3)))
    oo = Constant(0)
    m1 = variable(m1)
    m2 = variable(m2)
    m3 = variable(m3)
    mm = as_vector((m1, m2, m3))
    m_cryst = dot(at,mm)
    w_an = -kku*dot(mm,nu)**2 + kkp*dot(mm,np)**2 + kkc*(m_cryst[0]**2*m_cryst[1]**2 + m_cryst[0]**2*m_cryst[2]**2 + m_cryst[1]**2*m_cryst[2]**2) + 2*math.pi*M_s**2*m1**2
    an_vec = as_vector((-diff(w_an,m1)/2/kkp, -diff(w_an,m2)/2/kkp, -diff(w_an,m3)/2/kkp))
    #g_vec = as_vector((grad(dot(m,e_f))[0],grad(dot(m,e_f))[1],oo))
    #phi_vec = as_vector((-phi.dx(0), -phi.dx(1), oo))
    return an_vec # + hd_s

def hs_rest(m,p,e_f,phi):
    m1, m2, m3 = split(m)
    e1, e2, e3 = split(e_f)
    oo = Constant(0)
    vec = as_vector((oo, oo, m3)) #нет производной по третьей координатe от поля
    #g_vec = as_vector((grad(dot(m,e_f))[0],grad(dot(m,e_f))[1],oo))
    phi_vec = as_vector((-phi.dx(0), -phi.dx(1), oo))
    return vec + phi_vec

def grad3(v):
    oo = Constant(0)
    #vec = as_vector((grad(v)[0],grad(v)[1],oo))
    vec = as_vector((v.dx(0),v.dx(1),oo))
    return vec

def my_Hd_v(phi, m, m_z_bl):
    PI = Constant(math.pi)
    m1, m2, m3 = split(m)
    #oo = Constant(0)
    #vec = as_vector((grad(v)[0],grad(v)[1],oo))
    vec = as_vector((-phi.dx(0), -phi.dx(1), -4*PI*2*(m3-m_z_bl)))
    return vec

def to_2d(v):
    v1, v2, v3 = split(v)
    vec = as_vector((v1,v2))
    return vec

def dot_v(m,mm,w):      # магнитоэлектрические слагаемые убраны
    #m1, m2, m3 = split(m)
    mm1, mm2, mm3 = split(m)
    #e1, e2, e3 = split(e_f)
    #w1, w2, w3 = split(w)
    expr = dot(grad(cross(w,m)[0]),grad(mm1)) + \
        dot(grad(cross(w,m)[1]),grad(mm2)) + \
            dot(grad(cross(w,m)[2]),grad(mm3))
    return expr

def dots_v(m,mm,w,pp,e_f):
    #m1, m2, m3 = split(m)
    mm1, mm2, mm3 = split(m)
    e1, e2, e3 = split(e_f)
    #w1, w2, w3 = split(w)
    expr = dot(grad(cross(w,m)[0]),grad(mm1)) + \
        dot(grad(cross(w,m)[1]),grad(mm2)) + \
            dot(grad(cross(w,m)[2]),grad(mm3))
    return expr

def g_c(m,w,i):
    oo = Constant(0)
    expr = as_vector((cross(w,m)[i].dx(0),cross(w,m)[i].dx(1),oo))
    return expr

def mgm(m,pp,e_f,i):
    m1, m2, m3 = split(m)
    e1, e2, e3 = split(e_f)
    mm = [m1,m2,m3]
    E = [e1,e2,e3]
    oo = Constant(0)
    expr = as_vector((mm[i].dx(0),mm[i].dx(1),oo)) +2*pp*E[i]*m
    return expr

def dmdn(m,n):
    m1, m2, m3 = split(m)
    v1 = dot(grad(m1),n)
    v2 = dot(grad(m2),n)
    v3 = dot(grad(m3),n)
    return as_vector((v1,v2,v3))

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

alpha1 = 0.8 #0.9 #0.1 #0.0001 
alpha2 = 10   #parameter alpha
UU0 = 0*1.6*10/3 #Voltage (CGS)
AA = 9.5*10**(-8) #4.3e-6 #2*10**(-8) #(erg/cm) - exchange constant

# # Образец 27
# kku = 852.9 # erg/cm**3 - unaxial anisotropy constant
# kkp = 2440 # erg/cm**3 - rombic anisotropy constant
# kkc = -1*1275 # cubic anisotropy constant
#M_s = 4.26

# Образец 30
# kku = -312 # erg/cm**3 - unaxial anisotropy constant
# kkp = 3338 # erg/cm**3 - rombic anisotropy constant
# kkc = -1*1555 # cubic anisotropy constant
# M_s = 4.95

# Образец 32
kku = 1054 # erg/cm**3 - unaxial anisotropy constant
kkp = 1830 # erg/cm**3 - rombic anisotropy constant
kkc = -1*1016 # cubic anisotropy constant
M_s = 3.46

theta_0 = 0*math.pi/4

rr0 = 0.00003 # cm - effective electrode radius
#dd = math.sqrt(AA/kku)# characteristic domain wall width
beta = math.sqrt(1+2*math.pi*M_s**2/kku)
#beta_n = math.sqrt(1-(kkp-2*math.pi*M_s**2)/kku)
#dd_n = math.sqrt(AA/(kku+2*math.pi*M_s**2))
g = 10**(-6) # magnetoelectric constant
#p = g*UU0/rr0/(2*math.sqrt(AA*kk))
# p = g*UU0/1e-4/(2*math.sqrt(AA*kk)/6)*0.1
Hy = -30
xx0 = 0
yy0 = 2

L = 20

### Описание анизотропии
# # Образец 27
# tu = 40/180*math.pi
# psu = 189.4/180*math.pi
# psp = 9.44/180*math.pi

# Образец 30
# tu = 50.2/180*math.pi
# psu = -203/180*math.pi
# psp = -185/180*math.pi

# Образец 32
tu = 59.7/180*math.pi
psu = -16.7/180*math.pi
psp = 9.7/180*math.pi

betta_deg = 180
betta = betta_deg/180*np.pi

Nu = as_vector((sin(tu)*cos(psu), sin(tu)*sin(psu), cos(tu)))
NNu = np.array([np.sin(tu)*np.cos(psu), np.sin(tu)*np.sin(psu), np.cos(tu)])

sin_g = -cos(tu)/sqrt(cos(tu)**2 + sin(tu)**2*cos(psu-psp)**2)
cos_g = sin(tu)*cos(psu-psp)/sqrt(cos(tu)**2 + sin(tu)**2*cos(psu-psp)**2)
Np = as_vector((sin_g*cos(psp), sin_g*sin(psp), cos_g))

Sin_g = -np.cos(tu)/np.sqrt(np.cos(tu)**2 + np.sin(tu)**2*np.cos(psu-psp)**2)
Cos_g = np.sin(tu)*np.cos(psu-psp)/np.sqrt(np.cos(tu)**2 + np.sin(tu)**2*np.cos(psu-psp)**2)
NNp = np.array([Sin_g*np.cos(psp), Sin_g*np.sin(psp), Cos_g])

a_rot = as_matrix(((cos(betta),  -sin(betta), 0),
                (sin(betta), cos(betta), 0),
                (0, 0, 1)))

Nu = dot(a_rot, Nu)
Np = dot(a_rot, Np)

aa_rot = np.array([[ np.cos(betta),  -np.sin(betta), 0.],
              [np.sin(betta),  np.cos(betta), 0.],
              [0., 0., 1.]])

NNu = np.matmul(aa_rot, NNu)
NNp = np.matmul(aa_rot, NNp)

'''
Матрица, содержащая в строках разложение ортов рабочей СК Лисовского по ортам 
кристаллографической СК (для FEniCS)
'''
a0 = as_matrix(((-2/math.sqrt(5), 1/math.sqrt(5), 0),
                (0, 0, 1),
                 (1/math.sqrt(5), 2/math.sqrt(5), 0)))

'''
Матрица, содержащая в строках разложение ортов рабочей СК Лисовского по ортам 
кристаллографической СК
'''
Mat = np.array([[-2/np.sqrt(5), 1/np.sqrt(5), 0.],
               [0., 0., 1],
               [1/np.sqrt(5), 2/np.sqrt(5), 0.]])
'''
Матрица, переводящая компоненты намагниченности относительно текущей СК в 
компоненты намагниченности относительно кристаллографической СК
'''
Mat_1 = np.matmul(np.linalg.inv(Mat),np.linalg.inv(aa_rot))
#Mat_1 = np.identity(3)

from ufl.operators import transpose, inv

at = dot(inv(a0),inv(a_rot))

aa_res = aa_min(NNu, NNp, Mat_1, kku, kkp, kkc)
th_0, ph_0 = aa_res.x
print(th_0)
print(ph_0)
###
# In[part 1: symbolic]
sp.init_printing()
th, ph, y, d = sp.symbols('th_0 ph_0 x[0] d')
mat1 = sp.Matrix([[ sp.cos(sp.pi/2-ph),  sp.sin(sp.pi/2-ph), 0.],
              [-sp.sin(sp.pi/2-ph),  sp.cos(sp.pi/2-ph), 0.],
              [0., 0., 1]])
mat2 = sp.Matrix([[ 1,  0., 0.],
              [0., sp.cos(th), sp.sin(th)],
              [0., -sp.sin(th), sp.cos(th)]])

mat = mat1*mat2
#mat = mat1*mat2
v_0 = sp.Matrix([0, sp.sin(2*sp.atan(sp.exp(y))), sp.cos(2*sp.atan(sp.exp(y)))])
#v_0 = sp.Matrix([0, 0, 1])
v_1 = mat*v_0

v_1_x = sp.ccode(v_1[0])
v_1_y = sp.ccode(v_1[1])
v_1_z = sp.ccode(v_1[2])

# In[part 2]

mesh = IntervalMesh(1000, -L, L)

Elv = VectorElement('CG', mesh.ufl_cell(), 1, dim = 3)
FS = FunctionSpace(mesh, Elv)

v = Function(FS) #TrialFunction
w = TestFunction(FS)

def l_boundary(x, on_boundary):
    tol = 1E-16
    return (on_boundary and near(x[0],-L/2,tol))

def r_boundary(x, on_boundary):
    tol = 1E-16
    return (on_boundary and near(x[0],L/2,tol))

def boundary(x, on_boundary):
    return on_boundary


wall_type = 'bloch'# 'bloch'  'neel' 'h'
# Define boundary condition
if wall_type =='neel':
    ub = Expression(("0", "-sin(2*atan(exp(x[0]/d)))", "cos(2*atan(exp(x[0]/d)))"), degree = 4, d=1)
    #ub = Expression(("0", "-sin(2*atan(exp(x[1]/d)))*cos(a) + cos(2*atan(exp(x[1]/d)))*sin(a)", "sin(a)*sin(2*atan(exp(x[1]/d))) + cos(2*atan(exp(x[1]/d)))*cos(a)"), degree = 4, d=1/beta, a = theta_0)
    #ub_n = Expression(("0", "sin(2*atan(exp(x[1]/d))+a)", "cos(2*atan(exp(x[1]/d))+a)"), degree = 4, d=1/beta, a = theta_0)
    #ub = Expression(("0", "sqrt(1-(tanh(x[1]/d)*tanh(x[1]/d)))", "tanh(x[1]/d)", "0"), degree = 4, d=1/beta)
    
if wall_type =='bloch':
    ub = Expression((v_1_x, v_1_y, v_1_z), degree = 4, ph_0 = ph_0, th_0 = th_0)
    #m_bloch = project(Expression(("sin(2*atan(exp(x[1]/d)))", "0", "cos(2*atan(exp(x[1]/d)))"), degree = 4, d=1),FS)
    #ub = Expression(("sin(-2*atan(exp((x[1]-5)/d)) - 2*atan(exp((x[1]+5)/d)))", "0", "cos(2*atan(exp((x[1]-5)/d)) - 2*atan(exp((x[1]+5)/d)))"), degree = 4, d=1)
    #ub = Expression(("sin(3*x[1]/30)", "0", "cos(3*x[1]/30)"), degree = 2)
    #ub = Expression(("sqrt(1-(tanh(x[1]/d)*tanh(x[1]/d)))", "0", "tanh(x[1]/d)"), degree = 5, d=1)
#ub = Expression(("cos(x[1])", "0", "sin(x[1])", "0"), degree = 4)
if wall_type =='h':
    ub = Expression(("0", "0.", "1."), degree = 4)

BC = DirichletBC(FS, ub, boundary)

# route to files
betta_s = str(betta_deg)
route_0 = '/media/mnv/A2E41E9EE41E74AF/S32_min_new_theta_u/'
route = route_0 + betta_s +'_s/'

if not os.path.exists(route):
    os.makedirs(route)

#time_old = TimeSeries('/media/mnv/A2E41E9EE41E74AF/series_old_1d/m')
time_new = TimeSeries(route + 'series_new_1d/m')

in_type = 'new'
if in_type == 'old':
    t = 812.4000000000477
    m = Function(FS)
    time_old.retrieve(m.vector(), t)
if in_type == 'new':
    m = project(ub,FS)
if in_type == 'rand':
    m = project(ub,FS)
    m = DD_Hd.rand_vec(m, 0.001)
    m = norm_sol_s(m,FS)

dt = 0.1 #0.1 #0.025 ## 0.01
Dt = Constant(dt)
al = Constant(alpha1)
ku = Constant(kku)
kp = Constant(kkp)
kc = Constant(kkc)
Ms = Constant(M_s)
theta = 1
E_old = 0
th = Constant(theta)

m1, m2, m3 = split(m)

F = dot(w,(v-m)/Dt-al*cross(m,(v-m)/Dt))*dx \
+ (1-th)**2*dot(w,cross(m,h_rest(m, ku, kp, kc, M_s, Nu, Np, at)))*dx  + (1-th)*th*dot(w,cross(v,h_rest(m, ku, kp, kc, M_s, Nu, Np, at)))*dx + (1-th)*th*dot(w,cross(m,h_rest(v, ku, kp, kc, M_s, Nu, Np, at)))*dx + th**2*dot(w,cross(v,h_rest(v, ku, kp, kc, M_s, Nu, Np, at)))*dx \
    - (1-th)**2*dot_v(m,m,w)*dx - (1-th)*th*dot_v(m,v,w)*dx - (1-th)*th*dot_v(v,m,w)*dx - th**2*dot_v(v,v,w)*dx 

Jac = derivative(F,v)
diffr = Function(FS)
Hd = Function(FS)

title = 't' + ', '  + 'w_ex' + ', '  + 'w_a' + ', ' + 'w_a_u' + ', ' + 'w_a_p' + ', ' + 'w_a_c' + ', ' + 'w_a_s' + ', ' + 'w_total'  +  ', ' + 'diff\n'
file_txt = open(route + 'avg_table_1d.txt','w') # old version: '/media/mnv/A2E41E9EE41E74AF/avg_table_1d.txt'
file_txt.write(title)
file_txt.close()

vtkfile_m = File(route + 'graphs_1d/m.pvd')
vtkfile_diff = File(route + 'graphs_1d/diff.pvd')

i = 0
j = 0
count = 0
T =  1.
tol = 5E-8

N_f = 2000 #1000 #100000
n = FacetNormal(mesh)
# to def: ub, min w_a
while j <= 10:
    if i>=N_f:
        print(N_f, ' iterations reached')
        break
    #phi = DD_Hd.pot(m, wall_type, beta)
    #- M_s/2/k*Hy*w2*dx
    #- M_s/2/k*M_s*hy*w2*dx
    #- k_o/k*(v1-m1)*w1*dx
    #+ M_s*M_s/2/k*(w1*phi.dx(0) + w2*phi.dx(1))*dx
        # F = dot(w,(v-m)/Dt-al*cross(m,(v-m)/Dt))*dx \
        # + (1-th)**2*dot(w,cross(m,h_rest(m,pp,e_f)))*dx  + (1-th)*th*dot(w,cross(v,h_rest(m,pp,e_f)))*dx + (1-th)*th*dot(w,cross(m,h_rest(v,pp,e_f)))*dx + th**2*dot(w,cross(v,h_rest(v,pp,e_f)))*dx \
        #     - (1-th)**2*dot_v(m,m,w,pp,e_f)*dx - (1-th)*th*dot_v(m,v,w,pp,e_f)*dx - (1-th)*th*dot_v(v,m,w,pp,e_f)*dx - th**2*dot_v(v,v,w,pp,e_f)*dx \
        #         + 2*pp*dot(w,cross(m_b,e_f))*dot(to_2d(m_b),n)*ds
    solve(F==0,v, BC,J=Jac) # BC!!!
    
    v = norm_sol_s(v, FS)
    V = v.vector()
    M = m.vector()
    Diffr = V - M
    diffr.vector()[:] = Diffr/(L*dt)
    #Hd_v = project(-grad3(phi),FS)
    #cr = project(cross(m,dmdn(m,n)),FS)
    #Hd.vector()[:] = Hd_v.vector() + hd_s.vector() + hd_ext.vector()
    error = (m-v)**2*dx
    E = sqrt(abs(assemble(error)))/(L)/dt
    
    w_ex = assemble((dot(grad(m1),grad(m1)) + dot(grad(m2),grad(m2)) + dot(grad(m3),grad(m3)))*dx)/(L)
    m_cryst = dot(at,m)
    #w_an = -kku*dot(mm,nu)**2 + kkp*dot(mm,np)**2 + kkc*(m_cryst[0]**2*m_cryst[1]**2 + m_cryst[0]**2*m_cryst[2]**2 + m_cryst[1]**2*m_cryst[2]**2)
    w_a = assemble((-kku*dot(m,Nu)**2 + kkp*dot(m,Np)**2 + kkc*(m_cryst[0]**2*m_cryst[1]**2 + m_cryst[0]**2*m_cryst[2]**2 + m_cryst[1]**2*m_cryst[2]**2))*dx + 2*math.pi*M_s**2*m1**2*dx)/(kkp*L) #assemble(-m3*m3*dx)/(Lx*Ly)
    w_a_u = assemble((-kku*dot(m,Nu)**2)*dx)/(kkp*L)
    w_a_p = assemble((kkp*dot(m,Np)**2)*dx)/(kkp*L)
    w_a_c = assemble(((kkc+1)*(m_cryst[0]**2*m_cryst[1]**2 + m_cryst[0]**2*m_cryst[2]**2 + m_cryst[1]**2*m_cryst[2]**2))*dx)/(kkp*L) #!!!
    w_a_s = assemble((2*math.pi*M_s**2*m1**2)*dx)/(kkp*L)
    w_total = w_a+w_ex
    #w_hd = assemble(-dot(to_2d(m),-grad(phi))*dx)/(L)
    #w_hd_2 = assemble(-dot(m,hd_ext+Hd_v_y)*dx)/(Lx*Ly)
    #w_me = assemble(pp*dot(e_f,m*div(to_2d(m)) - grad(m)*m)*dx)/(Lx*Ly)
    data_ex = str(w_ex)
    data_a = str(w_a)
    data_a_u = str(w_a_u)
    data_a_p = str(w_a_p)
    data_a_c = str(w_a_c)
    data_a_s = str(w_a_s)
    data_total = str(w_total)
    #data_hd_1 = str(w_hd_1)
    #data_hd_2 = str(w_hd_2)
    #data_me = str(w_me)
    data = str(round(T,5)) + ', ' + data_ex + ', ' + data_a + ', ' + data_a_u + ', ' + data_a_p + ', ' + data_a_c + ', ' + data_a_s + ', ' + data_total + ', '  + str(E) + '\n'
    if i%1 == 0:
        vtkfile_m << (m, T)
        #vtkfile_hd_v << (phi, T)
        #vtkfile_hd_s << hd_s
        vtkfile_diff << (diffr, T)
        file_txt = open(route + 'avg_table_1d.txt','a')
        file_txt.write(data)
        file_txt.close()
        #vtkfile_cr << cr
        
    # vtkfile_m2 << m2
    # vtkfile_m3 << m3
    # vtkfile_l << u_l
    #plot(u3)
    
    v1, v2, v3 = v.split()
    # P = project(m*(m1.dx(0) + m2.dx(1)) - as_vector((m1*m1.dx(0)+m2*m1.dx(1), m1*m2.dx(0)+m2*m2.dx(1), m1*m3.dx(0)+m2*m3.dx(1))), FS_3)
    # vtkfile_P << P
    # error = (m-v)**2*dx
    # E = sqrt(abs(assemble(error)))/(Lx*Ly)/dt
    delta_E = E-E_old
    E_old = E
    print('delta = ', E, ', ', 'i = ', i)
    if E <= tol:
        j += 1
    i += 1
    
    if (abs(delta_E/E) <= 5E-3) and (delta_E < 0):
        count += 1
    else:
        count = 0
    if count >= 50:
        count = 0
        # dt = round(dt + 0.01, 4) #0.05
        # Dt.assign(dt)
        print('NEW Time Step:', dt)
    
    m.assign(v)
    #phi_n = DD_Hd.pot(m, wall_type, beta, phi, m_b_2d, pbc)
    #hd_s_n = DD_Hd.s_chg(m3, SL_space, FS_1, FS_3_1, FS_3_1, FS, idx, space_top, slp_pot, trace_space, trace_matrix)
    #phi.assign(phi_n)
    #hd_s.assign(hd_s_n)
    # U = u.vector()
    # m.vector()[:] = U
    m1, m2, m3 = m.split()
    T = T + dt

plot(v3)
vtkfile_m << (m, T)
vtkfile_diff << (diffr, T)

file_txt = open(route + 'avg_table_1d.txt','a')
file_txt.write(data)
file_txt.close()

angle_data = betta_s + ', ' + data_total + '\n'

angle_file_txt = open(route_0 + 'angle_w_tot.txt','a')
angle_file_txt.write(angle_data)
angle_file_txt.close()

time_new.store(m.vector(),i)
print(i)