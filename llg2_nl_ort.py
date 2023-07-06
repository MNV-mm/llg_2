# In[1]:


from fenics import *
import DD_Hd
import funcs_2
import numpy as np
import math
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
    #v2d = vertex_to_dof_map(FS)
    #d2v = dof_to_vertex_map(FS)
    u_array_2 = u_array#[v2d]
    i = 0
    while i+2 < N:
        norm = math.sqrt(math.pow(u_array_2[i],2) + math.pow(u_array_2[i+1],2) + math.pow(u_array_2[i+2],2))
        u_array_2[i] /= norm
        u_array_2[i+1] /= norm
        u_array_2[i+2] /= norm
        i += 3
    
    u_array = u_array_2#[d2v]
    u.vector()[:] = u_array
    return u

def max_norm(u):
    '''
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
    '''
    u1, u2, u3 = u.split()
    V1 = u1.compute_vertex_values()
    V2 = u2.compute_vertex_values()
    V3 = u3.compute_vertex_values()
    norm_prev = np.max(np.sqrt(V1*V1 + V2*V2 + V3*V3))
    return norm_prev

def h_rest(m,p, e_f, dedz, phi, hd_s, kku, kkp, kkc, nu, np, at):
    m1, m2, m3 = split(m)
    e1, e2, e3 = split(e_f)
    dedz_1, dedz_2, dedz_3 = split(dedz)
    ## есть маг поле по y
    vec = as_vector((-p*(2*e1*m1.dx(0) + 2*e2*m2.dx(0) + 2*e3*m3.dx(0) + m1*e1.dx(0) + m2*e2.dx(0) + m3*e3.dx(0) + m1*e1.dx(0) + m2*e1.dx(1) + m3*dedz_1), \
                     -p*(2*e1*m1.dx(1) + 2*e2*m2.dx(1) + 2*e3*m3.dx(1) + m1*e1.dx(1) + m2*e2.dx(1) + m3*e3.dx(1) + m1*e2.dx(0) + m2*e2.dx(1) + m3*dedz_2), \
                          -p*(m1*e3.dx(0) + m2*e3.dx(1) + m3*dedz_3 + m1*dedz_1 + m2*dedz_2 + m3*dedz_3)))
    oo = Constant(0)
    m1 = variable(m1)
    m2 = variable(m2)
    m3 = variable(m3)
    mm = as_vector((m1, m2, m3))
    m_cryst = dot(at,mm)
    w_an = -kku*dot(mm,nu)**2 + kkp*dot(mm,np)**2 + kkc*(m_cryst[0]**2*m_cryst[1]**2 + m_cryst[0]**2*m_cryst[2]**2 + m_cryst[1]**2*m_cryst[2]**2)
    an_vec = as_vector((-diff(w_an,m1)/2/kkp, -diff(w_an,m2)/2/kkp, -diff(w_an,m3)/2/kkp))
    #g_vec = as_vector((grad(dot(m,e_f))[0],grad(dot(m,e_f))[1],oo))
    phi_vec = as_vector((-phi.dx(0), -phi.dx(1), oo))
    return vec + an_vec + phi_vec #+ hd_s

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

def dot_v(m,mm,w,pp,e_f):
    #m1, m2, m3 = split(m)
    mm1, mm2, mm3 = split(m)
    e1, e2, e3 = split(e_f)
    #w1, w2, w3 = split(w)
    expr = dot(grad(cross(w,m)[0]),grad(mm1) + 2*pp*e1*to_2d(mm)) + \
        dot(grad(cross(w,m)[1]),grad(mm2) + 2*pp*e2*to_2d(mm)) + \
            dot(grad(cross(w,m)[2]),grad(mm3) + 2*pp*e3*to_2d(mm))
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

def mwrite(filename, out, a_type, rank):
    MPI.barrier(comm)
    if rank == 0:
        file_txt = open(filename,a_type)
        file_txt.write(out)
        file_txt.close()

comm = MPI.comm_world

rank = comm.Get_rank()
size = comm.Get_size()

alpha1 = 3  
alpha2 = 10   #parameter alpha
UU0 = (2*2*10/3) #Voltage (CGS)
AA = 9.5*10**(-8) #4.3e-6 #2*10**(-8) #(erg/cm) - exchange constant

# # Образец 27
# kku = 852.9 # erg/cm**3 - unaxial anisotropy constant
# kkp = 2440 # erg/cm**3 - rombic anisotropy constant
# kkc = -1*1275 # cubic anisotropy constant
#M_s = 4.26

# Образец 30
kku = -312 # erg/cm**3 - unaxial anisotropy constant
kkp = 3338 # erg/cm**3 - rombic anisotropy constant
kkc = -1*1555 # cubic anisotropy constant
M_s = 4.95

# Образец 32
# kku = 1054 # erg/cm**3 - unaxial anisotropy constant
# kkp = 1830 # erg/cm**3 - rombic anisotropy constant
# kkc = -1*1016 # cubic anisotropy constant
# M_s = 3.46

route_0 = '/home/mnv/llg_nl/'

theta_0 = 0*math.pi/4

rr0 = 0.00003 # cm - effective electrode radius
dd = math.sqrt(AA/kkp)# characteristic domain wall width
beta = math.sqrt(1+2*math.pi*M_s**2/kku)
#beta_n = math.sqrt(1-(kkp-2*math.pi*M_s**2)/kku)
dd_n = math.sqrt(AA/(kkp+2*math.pi*M_s**2))
g = 10**(-6) # magnetoelectric constant
#p = g*UU0/rr0/(2*math.sqrt(AA*kk))
# p = g*UU0/1e-4/(2*math.sqrt(AA*kk)/6)*0.1
Hy = -30
xx0 = 0
yy0 = 2
#beta = 1.2#parameter beta
#print(parameters.linear_algebra_backend)
#list_linear_solver_methods()

# In[anisotropy]
# # Образец 27
# tu = 40/180*math.pi
# psu = 189.4/180*math.pi
# psp = 9.44/180*math.pi

# Образец 30
tu = 50.2/180*math.pi
psu = -203/180*math.pi
psp = -185/180*math.pi

# Образец 32
# tu = 46.2/180*math.pi
# psu = -16.7/180*math.pi
# psp = 9.7/180*math.pi

betta_deg = 90 + 32 #(учтено, что здесь нормаль к плоскости стенки направлена вдоль оси Оy)
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

print('before aa_min')

#aa_res = DD_Hd.aa_min(NNu, NNp, Mat_1, kku, kkp, kkc)
#th_0, ph_0 = aa_res.x
th_0 = 2.013269620451453
ph_0 = 3.093092175684844
print(th_0)
print(ph_0)


# In[2]:


# Create mesh and define function space
Lx = 570 # 60 150 80
Ly = 200 # 30 80 40
#DD_Hd.pe_EF(5,30,1,Lx,Ly,'/home/mnv/Documents/python_doc/llg_nl/E_series')

#FS_1, FS_3, FS_3_1, FS, e_v = DD_Hd.pe_EF(5,30,1,Lx,Ly)
#mesh = FS.mesh()
mesh = RectangleMesh(Point(-Lx/2,-Ly/2), Point(Lx/2,Ly/2), 4*570, 4*200) # 1140, 400
mesh_0 = Mesh(route_0 + 'MESH.xml')
#mesh_0 = Mesh()

hdf_E = HDF5File(mesh.mpi_comm(), route_0 + 'results/e_field/E_hdf_-15_big.h5', 'r')
#hdf_E.read(mesh_0, "/my_mesh")

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < (-Lx/2 + DOLFIN_EPS) and x[0] > (-Lx/2 - DOLFIN_EPS) and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - Lx
        y[1] = x[1]

pbc = PeriodicBoundary()

SL_mesh = RectangleMesh(Point(-Lx/2,-Ly/2),Point(Lx/2,Ly/2),int(2*Lx),int(2*Ly))

z_max = 0.5
p1 = Point(-Lx/2,-Ly/2,-z_max)
p2 = Point(Lx/2,Ly/2,z_max)
nx = 570
ny = 200
#mesh_3d = BoxMesh(p1,p2,nx,ny,2)

#SL_space, FS_1, FS_3, FS_3_1, FS

El = VectorElement('CG', triangle, 1, dim=3)
FS_0 = FunctionSpace(mesh_0, El, constrained_domain=pbc)

FS = FunctionSpace(mesh, El, constrained_domain=pbc)

#El_1 = FiniteElement('CG', triangle, 1)
#FS_1 = FunctionSpace(mesh, El_1)

#SL_El = FiniteElement('CG', triangle, 1)
#SL_space = FunctionSpace(SL_mesh, SL_El)

#El_3 = FiniteElement('CG', tetrahedron, 2)
#FS_3 = FunctionSpace(mesh_3d, El_3)

#El_3_1 = FiniteElement('CG', tetrahedron, 1)
#FS_3_1 = FunctionSpace(mesh_3d, El_3_1)

e_v_0 = Function(FS_0)
dedz_v_0 = Function(FS_0)

#E_series = TimeSeries(route_0 + 'results/e_field/E_mid_15_small')
#dEdz_series = TimeSeries(route_0 + 'results/e_field/E_mid_15_small_dEdz')

#E_series.retrieve(e_v_0.vector(),0)
#dEdz_series.retrieve(dedz_v_0.vector(),0)

hdf_E.read(e_v_0, "/e_field")
hdf_E.read(dedz_v_0, "/dedz_field")
hdf_E.close()

#e_v = interpolate(e_v_0, FS)
#dedz_v = interpolate(dedz_v_0, FS)

e_v = Function(FS)
dedz_v = Function(FS)

LagrangeInterpolator.interpolate(e_v, e_v_0)
LagrangeInterpolator.interpolate(dedz_v, dedz_v_0)

E_array = e_v.vector().get_local()
E_max = MPI.max(comm, max_norm(e_v))

print("E_max = ", E_max)

dEdz_array = dedz_v.vector().get_local()

e_v.vector()[:] = E_array/E_max
dedz_v.vector()[:] = dEdz_array/E_max

p = g*UU0/math.sqrt(AA/kkp)/(2*math.sqrt(AA*kkp))*(2*E_max) # 2E_max = 0.01

print("p = ", p)

dedz_1, dedz_2, dedz_3 = split(dedz_v)

#El = VectorElement('CG', triangle, 1, dim=3)
El1 = FiniteElement('CG', triangle, 1)
#FS = FunctionSpace(mesh, El)
FS_1 = FunctionSpace(mesh,El1)
# dy = 5
# R0 = 10
# s_s = 0.05 #0.1
# s_L = 0.1 #0.9
# lax = 10 #adsorbtion layer thickness
# lay = 10
# nx = 330 #300
# ny = 150#400
# p1 = Point(-Lx/2,-Ly/2)
# p2 = Point(Lx/2,Ly/2)
# mesh = RectangleMesh(p1,p2,nx,ny)
#mesh = DD_Hd.wall_mesh(Lx,Ly,dy,R0,s_s,s_L)
v = Function(FS) #TrialFunction
w = TestFunction(FS)
#K = FunctionSpace(mesh,El2)
##########################hd_side = DD_Hd.side_pot(FS_1, FS_3_1, FS_3_1, FS, 50, z_max, Lx, Ly, 240)

# In[] # Symbolic expressions
# x, y, z = sp.symbols('x y z')
# xx, yy = sp.symbols('x[0] x[1]')
# x0, y0 = sp.symbols('x0 y0')
# d, r0, U0 = sp.symbols('d r0 U0')

# f_expr = U0*r0/sp.sqrt((r0-z)**2+((x-x0)**2 + (y-y0)**2))
# E1 = -sp.diff(f_expr,x)
# E1 = sp.simplify(E1.subs([(x,d*x),(y,d*y),(z,d*z),(x0,d*x0),(y0,d*y0),(z,0),(x,xx),(y,yy)])/U0*r0)
# E2 = -sp.diff(f_expr,y)
# E2 = sp.simplify(E2.subs([(x,d*x),(y,d*y),(z,d*z),(x0,d*x0),(y0,d*y0),(z,0),(x,xx),(y,yy)])/U0*r0)
# E3 = -sp.diff(f_expr,z)
# E3 = sp.simplify(E3.subs([(x,d*x),(y,d*y),(z,d*z),(x0,d*x0),(y0,d*y0),(z,0),(x,xx),(y,yy)])/U0*r0)
# E1_c=sp.ccode(E1)
# E2_c=sp.ccode(E2)
# E3_c=sp.ccode(E3)
# #print(E3_c)

# pe_p_str = DD_Hd.pe_p(1,5,20,1)
# pe_p_expr = Expression(pe_p_str, degree = 2)
# pe_p= project(pe_p_expr,FS_1)
# vtkfile_pe_p = File('graphs/pe_p.pvd')
# vtkfile_pe_p << pe_p

# u_0 = UU0
# aa = 5
# bb = 4*Ly
# cc = 1
# pe_ef_str = DD_Hd.pe_ef(aa,bb,cc)
# E_pe_expr = Expression((pe_ef_str[0],pe_ef_str[1],pe_ef_str[2]), degree = 4, a = aa, b = bb, c = cc, z = -20)
# #Ex = Expression(pe_ef_str[0],degree = 2, z = -1.1*c)
# E_pe = project(E_pe_expr,FS) #interpolate
# #E_pe_x = project(Ex,FS_1) 
# vtkfile_pe_ef = File('graphs/pe_ef.pvd')
# vtkfile_pe_ef << E_pe

# In[part 1: symbolic for BC]
sp.init_printing()
th, ph, y, d = sp.symbols('th_0 ph_0 x[1] d')
mat1 = sp.Matrix([[ sp.cos(sp.pi/2-ph),  sp.sin(sp.pi/2-ph), 0.],
              [-sp.sin(sp.pi/2-ph),  sp.cos(sp.pi/2-ph), 0.],
              [0., 0., 1]])
mat2 = sp.Matrix([[ 1,  0., 0.],
              [0., sp.cos(th), sp.sin(th)],
              [0., -sp.sin(th), sp.cos(th)]])

mat = mat1*mat2
#mat = mat1*mat2
v_0 = sp.Matrix([sp.sin(2*sp.atan(sp.exp(y))), 0, sp.cos(2*sp.atan(sp.exp(y)))])
#v_0 = sp.Matrix([0, 0, 1])
v_1 = mat*v_0

v_1_x = sp.ccode(v_1[0])
v_1_y = sp.ccode(v_1[1])
v_1_z = sp.ccode(v_1[2])
I1 = 2*sp.atan(sp.exp(y))
I2 = sp.asinh(3/4) + y - sp.asinh((2*sp.exp(y)+sp.exp(4*y))/(1+sp.exp(2*y))/2)
intg_0 = sp.Matrix([I1, 0, I2])

intg_1 = mat*intg_0
int_y = sp.ccode(intg_1[1]*4*sp.pi)

# In[3]:
wall_type = 'bloch'# 'bloch'  'neel' 'h'
# Define boundary condition
if wall_type =='neel':
    ub = Expression(("0", "-sin(2*atan(exp(x[1]/d)))", "cos(2*atan(exp(x[1]/d)))"), degree = 4, d=1)
    #ub = Expression(("0", "-sin(2*atan(exp(x[1]/d)))*cos(a) + cos(2*atan(exp(x[1]/d)))*sin(a)", "sin(a)*sin(2*atan(exp(x[1]/d))) + cos(2*atan(exp(x[1]/d)))*cos(a)"), degree = 4, d=1/beta, a = theta_0)
    #ub_n = Expression(("0", "sin(2*atan(exp(x[1]/d))+a)", "cos(2*atan(exp(x[1]/d))+a)"), degree = 4, d=1/beta, a = theta_0)
    #ub = Expression(("0", "sqrt(1-(tanh(x[1]/d)*tanh(x[1]/d)))", "tanh(x[1]/d)", "0"), degree = 4, d=1/beta)
    
if wall_type =='bloch':
    ub = Expression((v_1_x, v_1_y, v_1_z), degree = 4, ph_0 = ph_0, th_0 = th_0)
    #ub = Expression(("-sin(2*atan(exp(x[1]/d)))", "cos(2*atan(exp(x[1]/d)))*sin(a)", "cos(2*atan(exp(x[1]/d)))*cos(a)"), degree = 4, d=1, a = -theta_0)

if wall_type =='h':
    ub = Expression(("0", "0.", "1."), degree = 4)

phi_nl = Expression(int_y, degree = 4, ph_0 = ph_0, th_0 = th_0) # Expression("4*p*2*b*atan(exp(x[1]/b))", degree=4, p = math.pi, b = beta)
Hy_expr = Expression("-(5.5 + 0.00000002*(pow(x[1],6) + 300000*pow(x[1],2)))", degree = 4)

# Define electric field
electrode_type = 'plane' # 'plane'
if electrode_type == 'circle':
    e1 = Expression((E1_c),degree = 2, U0 = UU0, d = dd, r0 = rr0, x0 = xx0, y0 = yy0)   
    e2 = Expression((E2_c),degree = 2, U0 = UU0, d = dd, r0 = rr0, x0 = xx0, y0 = yy0)
    e3 = Expression((E3_c),degree = 2, U0 = UU0, d = dd, r0 = rr0, x0 = xx0, y0 = yy0)
    e_v = Expression((E1_c, E2_c, E3_c), degree = 2, U0 = UU0, d = dd, r0 = rr0, x0 = xx0, y0 = yy0)
if electrode_type == 'plane':
    print('plane_electrode')
    #e_v = Expression((pe_ef_str[0],pe_ef_str[1],pe_ef_str[2]), degree = 4, a = a, b = b, c = c, z = -1.1*c)

def boundary(x, on_boundary):
    return on_boundary

def my_boundary(x, on_boundary):
    tol = 1E-16
    return (on_boundary and near(x[1],Ly/2,tol)) or (on_boundary and near(x[1],-Ly/2,tol))

BC = DirichletBC(FS, ub, my_boundary)
# Define initial value
time_old = TimeSeries(route_0 + 'results/series_old/m')
time_new = TimeSeries(route_0 + 'results/series_new/m')

in_type = 'old'
if in_type == 'old':
    hdf_m_old = HDF5File(mesh.mpi_comm(), route_0 + 'results/m_old/m_final.h5', 'r')
    m = Function(FS)
    hdf_m_old.read(m, "/m_field")
    phi_0 = Function(FS_1)
    hdf_m_old.read(phi_0, "/demag_pot")
    hdf_m_old.close()
if in_type == 'new':
    m = project(ub,FS)
    #phi_0 = Function(FS_1)
    phi_0 = project(phi_nl, FS_1)
if in_type == 'rand':
    m = project(ub,FS)
    m = DD_Hd.rand_vec(m, 0.001)
    m = norm_sol_s(m,FS)
    phi_0 = project(phi_nl, FS_1)
    
m_b = project(ub,FS)
m = norm_sol_s(m, FS)
m1, m2, m3 = m.split()

h = 0.001 #cm
l = math.sqrt(4*math.sqrt(AA*kkp)/(M_s**2)*h)/dd
Z = h/dd/2

print('before_hds')

#idx, space_top, slp_pot, trace_space, trace_matrix = DD_Hd.s_chg_prep(SL_space, FS_1, FS_3_1, FS_3_1, FS, Z)
#hd_s = DD_Hd.s_chg(m3, SL_space, FS_1, FS_3_1, FS_3_1, FS, idx, space_top, slp_pot, trace_space, trace_matrix)
# vtkfile_hd_s = File('/media/mnv/A2E41E9EE41E74AF/graphs/hd_s.pvd')
# vtkfile_hd_s << hd_s

print('after_hds')

hd_ext_expr = funcs_2.n_pair(Ly, l, Z, 0, 4)
hd_ext = project(hd_ext_expr, FS)
#vtkfile_Hd_ext = File(route_0 + 'results/graphs/Hd_ext.pvd')
#vtkfile_Hd_ext << hd_ext
H_st = project(Expression(('0', '0', '-10/20*x[1]'), degree = 4),FS)
# vtkfile_hd_ext = File('/media/mnv/A2E41E9EE41E74AF/graphs/hd.pvd')
# vtkfile_hd_ext << hd_ext

e_f = e_v # project(e_v,FS)
m1, m2, m3 = split(m)
e1, e2, e3 = split(e_f)#e_f.split()
v1, v2, v3 = split(v) #split(v)
w1, w2, w3 = split(w)
# e3_values = e3.compute_vertex_values()
# m3_values = m3.compute_vertex_values()
al = Constant(alpha1)
# class A(Expression):
#     def set_al_values(self, Lx, Ly, lax, lay, al_0, al_1):
#         self.Lx, self.Ly = Lx, Ly
#         self.lax, self.lay = lax, lay
#         self.al_0, self.al_1 = al_0, al_1
        
#     def eval(self, value, x):
#         "Set value[0] to value at point x"
#         tol = 1E-14
#         if x[0] <= 1: #-Lx/2 + lax + tol:
#             value[0] = self.al_0
#         # elif x[0] >= Lx/2-lax +tol:
#         #     value[0] = self.al_0
#         # elif x[1] <= -Ly/2 + lay +tol:
#         #     value[0] = self.al_0
#         # elif x[1] >= Ly/2 - lay +tol:
#         #     value[0] = self.al_0
#         else:
#             value[0] = self.al_1

# al = A(degree = 0)
# al.set_al_values(Lx, Ly, lax, lay, alpha1, alpha2)
# tol = 1E-14
# al = Expression('(x[0] <= -65 + tol) || (x[0]>=65+tol) || (x[1]<=-30+tol) || (x[1]>=30+tol)? alpha2:alpha1', degree = 0, tol = tol, alpha2 = alpha2, alpha1 = alpha1)
pp = Constant(p)#p
ku = Constant(kku)
kp = Constant(kkp)
kc = Constant(kkc)
Ms = Constant(M_s)
hy = project(Hy_expr,FS_1)

#u_n = interpolate(ub, V)
#u_n = Function(V)
#u_n1, u_n2, u_n3 = split(u_n)
#/media/mnv/A2E41E9EE41E74AF/
m_file = XDMFFile(route_0 + "results/graphs/m.xdmf")
diff_file =  XDMFFile(route_0 + "results/graphs/diff.xdmf")
hd_v_file =  XDMFFile(route_0 + "results/graphs/hd_v.xdmf")

e_file =  XDMFFile(route_0 + 'results/graphs/e_file.xdmf')
e_file.write(e_f)
e_file.close()
# In[4]:
# In[5]
mx, my, mz = m.split()
m_b_1, m_b_2, m_b_3 = split(m_b)
m_b_2d = as_vector((m_b_1,m_b_2))

phi = DD_Hd.pot(m, wall_type, beta, phi_0, m_b_2d, pbc)
i = 0
j = 0
count = 0
dt = 0.001
Dt = Constant(dt)
T =  1
tol = 1E-8
theta = 1
E_old = 0
th = Constant(theta)
N_f = 150 #100000
n = FacetNormal(mesh)
oo = Constant(0)
PI = Constant(math.pi)
Hd_v_y = as_vector((oo, Constant(0.), oo)) #Constant(-26/2) on y axis
#hd_s+hd_ext
#F = dot(w,(v-m)/Dt-al*cross(m,(v-m)/Dt))*dx \
#+ (1-th)**2*dot(w,cross(m,h_rest(m,pp,e_f,dedz_v,M_s*M_s/2/kp*phi,M_s*M_s/2/kp*(hd_ext + Hd_v_y), ku, kp, kc, Nu, Np, at)))*dx  + (1-th)*th*dot(w,cross(v,h_rest(m,pp,e_f,dedz_v,M_s*M_s/2/kp*phi,M_s*M_s/2/kp*(hd_ext + Hd_v_y), ku, kp, kc, Nu, Np, at)))*dx + (1-th)*th*dot(w,cross(m,h_rest(v,pp,e_f,dedz_v,M_s*M_s/2/kp*phi,M_s*M_s/2/kp*(hd_ext + Hd_v_y), ku, kp, kc, Nu, Np, at)))*dx + th**2*dot(w,cross(v,h_rest(v,pp,e_f,dedz_v,M_s*M_s/2/kp*phi,M_s*M_s/2/kp*(hd_ext + Hd_v_y), ku, kp, kc, Nu, Np, at)))*dx \
    #- (1-th)**2*dot_v(m,m,w,pp,e_f)*dx - (1-th)*th*dot_v(m,v,w,pp,e_f)*dx - (1-th)*th*dot_v(v,m,w,pp,e_f)*dx - th**2*dot_v(v,v,w,pp,e_f)*dx 
    # \
    #     + dot(w,cross(m_b,dmdn(m_b,n)))*ds + 2*pp*dot(w,cross(m_b,e_f))*dot(to_2d(m_b),n)*ds
F = dot(w,(v-m)/Dt-al*cross(v,(v-m)/Dt))*dx + dot(w,cross(v,h_rest(v,pp,e_f,dedz_v,M_s*M_s/2/kp*phi,M_s*M_s/2/kp*(hd_ext + Hd_v_y), ku, kp, kc, Nu, Np, at)))*dx - dot_v(v,v,w,pp,e_f)*dx + dot(w,cross(m,dmdn(m,n)))*ds + 2*pp*dot(w,cross(m,e_f))*dot(to_2d(m),n)*ds
Jac = derivative(F,v)
diffr = Function(FS)
Hd = Function(FS)

title = 't' + ', '  + 'w_ex' + ', '  + 'w_a' + ', ' + 'w_a_u' + ', ' + 'w_a_p' + ', ' + 'w_a_c' + ', ' + 'w_hd_1' + ', ' + 'w_hd_2'  +  ', ' + 'w_me' + ', '  + 'w_tot' + ', '  + 'diff\n'
#file_txt = open(route_0 + 'results/avg_table.txt','w')
#file_txt.write(title)
#file_txt.close()
mwrite(route_0 + 'results/avg_table.txt', title, 'w', rank)
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
    solve(F==0, v, BC, J=Jac, solver_parameters = { "newton_solver": { "absolute_tolerance": 1e-12, "relative_tolerance": 1e-11}}) # BC!!!
    
    v = norm_sol_s(v, FS)
    V = v.vector()
    M = m.vector()
    Diffr = V - M
    diffr.vector()[:] = Diffr/(Lx*Ly*dt)
    diff_max = MPI.max(comm, np.max(np.abs(diffr.vector().get_local())))
    #Hd_v = project(-grad3(phi),FS)
    #cr = project(cross(m,dmdn(m,n)),FS)
    #Hd.vector()[:] = Hd_v.vector() + hd_s.vector() + hd_ext.vector()
    error = (m-v)**2*dx
    E = sqrt(abs(assemble(error)))/(Lx*Ly)/dt
    
    w_ex = MPI.sum(comm, assemble((dot(grad(m1),grad(m1)) + dot(grad(m2),grad(m2)) + dot(grad(m3),grad(m3)))*dx)/(Lx*Ly))
    m_cryst = dot(at,m)
    w_a = MPI.sum(comm, assemble((-kku*dot(m,Nu)**2 + kkp*dot(m,Np)**2 + kkc*(m_cryst[0]**2*m_cryst[1]**2 + m_cryst[0]**2*m_cryst[2]**2 + m_cryst[1]**2*m_cryst[2]**2))*dx)/(Lx*Ly*kkp))
    w_a_u = MPI.sum(comm, assemble((-kku*dot(m,Nu)**2)*dx)/(Lx*Ly*kkp)) #assemble(-m3*m3*dx)/(Lx*Ly)
    w_a_p = MPI.sum(comm, assemble((kkp*dot(m,Np)**2)*dx)/(Lx*Ly*kkp))
    w_a_c = MPI.sum(comm, assemble((kkc*(m_cryst[0]**2*m_cryst[1]**2 + m_cryst[0]**2*m_cryst[2]**2 + m_cryst[1]**2*m_cryst[2]**2))*dx)/(Lx*Ly*kkp))
    w_hd_1 = MPI.sum(comm, assemble(-dot(to_2d(m),-grad(phi))*dx)/(Lx*Ly)*(M_s*M_s/2/kkp))
    w_hd_2 = MPI.sum(comm, assemble(-dot(m,hd_ext+Hd_v_y)*dx)/(Lx*Ly)*M_s/2/kkp)
    w_me = MPI.sum(comm, assemble(-pp*dot(e_f,m*div(to_2d(m)) - grad(m)*m)*dx)/(Lx*Ly))
    w_tot = w_a + w_ex + w_hd_1 + w_me
    data_ex = str(w_ex)
    data_a = str(w_a)
    data_w_a_u = str(w_a_u)
    data_w_a_p = str(w_a_p)
    data_w_a_c = str(w_a_c)
    data_hd_1 = str(w_hd_1)
    data_hd_2 = str(w_hd_2)
    data_w_me = str(w_me)
    data_tot = str(w_tot)
    data = str(round(T,5)) + ', ' + data_ex + ', ' + data_a + ', ' + data_w_a_u + ', ' + data_w_a_p + ', ' + data_w_a_c + ', '  + data_hd_1 + ', ' + data_hd_2 + ', ' + data_w_me + ', ' + data_tot + ', ' + str(E) + '\n'
    #file_txt = open(route_0 + 'results/avg_table.txt','a')
    #file_txt.write(data)
    #file_txt.close()
    mwrite(route_0 + 'results/avg_table.txt', data, 'a', rank)
    if i%30 == 0:
        m_file.write(m, T)
        hd_v_file.write(phi, T)
        diff_file.write(diffr, T)
    T = T + dt    
    # vtkfile_m2 << m2
    # vtkfile_m3 << m3
    # vtkfile_l << u_l
    #plot(u3)
    
    v1, v2, v3 = v.split()
    #P = project(m*(m1.dx(0) + m2.dx(1)) - as_vector((m1*m1.dx(0)+m2*m1.dx(1), m1*m2.dx(0)+m2*m2.dx(1), m1*m3.dx(0)+m2*m3.dx(1))), FS_3)
    # error = (m-v)**2*dx
    # E = sqrt(abs(assemble(error)))/(Lx*Ly)/dt
    delta_E = E-E_old
    E_old = E
    print('delta = ', E, ', ', 'i = ', i)
    print('diff max = ', diff_max)
    if E <= tol:
        j += 1
    i += 1
    
    if (abs(delta_E/E) <= 1E-2):# and (delta_E < 0):
        count += 1
    else:
        count = 0
    if (count >= 10) and (dt <= 1):
        count = 0
        dt = 1*dt #0.05
        Dt.assign(dt)
        #alpha1 = 2*alpha1
        #al.assign(alpha1)
        print('NEW Time Step:', dt)
    
    m.assign(v)
    phi_n = DD_Hd.pot(m, wall_type, beta, phi, m_b_2d, pbc)
    #hd_s_n = DD_Hd.s_chg(m3, SL_space, FS_1, FS_3_1, FS_3_1, FS, idx, space_top, slp_pot, trace_space, trace_matrix)
    phi.assign(phi_n)
    #hd_s.assign(hd_s_n)
    # U = u.vector()
    # m.vector()[:] = U
    m1, m2, m3 = m.split()
    
    hdf_m = HDF5File(mesh.mpi_comm(), route_0 + 'results/series_new/m_final.h5', 'w')
    hdf_m.write(mesh, "/my_mesh")
    hdf_m.write(m, "/m_field")
    hdf_m.write(phi, "/demag_pot")
    hdf_m.close()
    


#plot(v3)
m_file.write(m, T)
m_file.close()
hd_v_file.write(phi, T)
hd_v_file.close()
diff_file.write(diffr, T)
diff_file.close()

mwrite(route_0 + 'results/avg_table.txt', data, 'a', rank)
#file_txt = open(route_0 + 'results/avg_table.txt','a')
#file_txt.write(data)
#file_txt.close()
hdf_m = HDF5File(mesh.mpi_comm(), route_0 + 'results/series_new/m_final.h5', 'w')
hdf_m.write(mesh, "/my_mesh")
hdf_m.write(m, "/m_field")
hdf_m.write(phi, "/demag_pot")
hdf_m.close()
print(i)
print("dt = ", dt)
# In[ ]:


