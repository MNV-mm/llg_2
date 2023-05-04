import bempp.api
import scipy
import numpy as np
import sympy as sp
import math
#import gmsh
import os
import meshio
import gc
from fenics import *
#stray field potential
def pot(m, wall_type, b, phi_prev, m_b, pbc):
    FS = m.function_space()
    mesh = FS.mesh()
    degree = FS.ufl_element().degree()
    #nFS = FunctionSpace(mesh,'CG',degree, constrained_domain=pbc)
    nFS = FunctionSpace(mesh,'CG', degree)
    
    u = TrialFunction(nFS)
    v = TestFunction(nFS)
    m1, m2, m3 = split(m)
    m_2d = as_vector((m1,m2))
    
    kk = 1000 #1054# erg/cm**3 - unaxial anisotropy constant
    M_s = 3.46
    t0 = 1*45/180*math.pi
    
    # vtkfile = File('/media/mnv/A2E41E9EE41E74AF/graphs/phi_prev.pvd')
    # vtkfile << phi_prev
    
    bv_bl = Expression('0',degree = 3)
    bv_nl = Expression("-4*p*(2*b*atan(exp(x[1]/b))*cos(t0) + x[1]*sin(t0) - b*asinh(1/2*(2*exp(2*x[1]/b) + exp(4*x[1]/b))/(1+exp(2*x[1]/b)))*sin(t0))", degree=4, p = math.pi, b = b, t0 = t0)
    #bv_nl = Expression("4*p*(2*b*atan(exp(x[1]/b))*sin(t0))", degree=4, p = math.pi, b = b, t0 = t0)
    def boundary(x, on_boundary):
        return on_boundary
    if wall_type == 'bloch':
        BC = DirichletBC(nFS, phi_prev, boundary)
        
    if wall_type == 'neel':
        BC = DirichletBC(nFS, phi_prev, boundary)
    
    f_pi = math.pi*4
    F_Pi = Constant(f_pi)
    n = FacetNormal(mesh)
    
    #F = dot(grad(u),grad(v))*dx + F_Pi*(m1.dx(0) + m2.dx(1))*v*dx - F_Pi*v*dot(n,m_b)*ds
    F =  -F_Pi*v*dot(m_b,n)*ds - dot(grad(v),grad(u))*dx + F_Pi*dot(m_2d,grad(v))*dx # -F_Pi*v*dot(m_b,n)*ds
    a = lhs(F)
    L = rhs(F)
    
    A = assemble(a)
    b = assemble(L)
    BC.apply(A,b)
    
    solver = KrylovSolver('gmres', 'hypre_euclid') #KrylovSolver('gmres', 'ilu') #hypre_euclid
    solver.parameters["nonzero_initial_guess"] = True
    u = Function(nFS)
    u.vector()[:] = phi_prev.vector()
    U = u.vector()
    
    solver.solve(A, U, b)
    u.vector()[:] = U
    
    return u
#My Mesh
def wall_mesh(Lx,Ly,dy,r0,s_s,s_L):
    gmsh.initialize()
    gmsh.model.add("wall1")
    gmsh.option.setNumber("Mesh.MshFileVersion", 2)

    gmsh.model.geo.addPoint(-Lx/2, -Ly/2, 0, s_L, 1)
    gmsh.model.geo.addPoint(-Lx/2, Ly/2, 0, s_L, 2)
    gmsh.model.geo.addPoint(Lx/2, Ly/2, 0, s_L, 3)
    gmsh.model.geo.addPoint(Lx/2, -Ly/2, 0, s_L, 4)

    gmsh.model.geo.addPoint(-Lx/2, -dy/2, 0, s_s, 5)
    gmsh.model.geo.addPoint(-Lx/2, dy/2, 0, s_s, 6)
    gmsh.model.geo.addPoint(Lx/2, dy/2, 0, s_s, 7)    
    gmsh.model.geo.addPoint(Lx/2, -dy/2, 0, s_s, 8)   

    gmsh.model.geo.addPoint(-r0/2, -Ly/2, 0, s_L, 9)  #s_s
    gmsh.model.geo.addPoint(-r0/2, -dy/2, 0, s_L, 10) #s_s
    gmsh.model.geo.addPoint(r0/2, -dy/2, 0, s_L, 11)  #s_s
    gmsh.model.geo.addPoint(r0/2, -Ly/2, 0, s_L, 12)  #s_s

    gmsh.model.geo.addPoint(-r0/2, dy/2, 0, s_L, 13)  #s_s
    gmsh.model.geo.addPoint(-r0/2, Ly/2, 0, s_L, 14)  #s_s
    gmsh.model.geo.addPoint(r0/2, Ly/2, 0, s_L, 15)   #s_s
    gmsh.model.geo.addPoint(r0/2, dy/2, 0, s_L, 16)   #s_s

    gmsh.model.geo.addLine(1, 5, 1)
    gmsh.model.geo.addLine(5, 6, 2)
    gmsh.model.geo.addLine(6, 2, 3)

    gmsh.model.geo.addLine(2, 14, 4)
    gmsh.model.geo.addLine(14, 15, 5)
    gmsh.model.geo.addLine(15, 3, 6)

    gmsh.model.geo.addLine(3, 7, 7)
    gmsh.model.geo.addLine(7, 8, 8)
    gmsh.model.geo.addLine(8, 4, 9)

    gmsh.model.geo.addLine(4, 12, 10)
    gmsh.model.geo.addLine(12, 9, 11)
    gmsh.model.geo.addLine(9, 1, 12)

    gmsh.model.geo.addLine(13, 6, 13)
    gmsh.model.geo.addLine(14, 13, 14)

    gmsh.model.geo.addLine(16, 13, 15)
    gmsh.model.geo.addLine(15, 16, 16)

    gmsh.model.geo.addLine(7, 16, 17)

    gmsh.model.geo.addLine(5, 10, 18)
    gmsh.model.geo.addLine(10, 9, 19)

    gmsh.model.geo.addLine(10, 11, 20)
    gmsh.model.geo.addLine(11, 12, 21)

    gmsh.model.geo.addLine(11, 8, 22)

    gmsh.model.geo.addCurveLoop([1, 18, 19, 12], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.geo.addCurveLoop([2, -13, -15, -17, 8, -22, -20, -18], 2)
    gmsh.model.geo.addPlaneSurface([2], 2)

    gmsh.model.geo.addCurveLoop([3, 4, 14, 13], 3)
    gmsh.model.geo.addPlaneSurface([3], 3)

    gmsh.model.geo.addCurveLoop([-14, 5, 16, 15], 4)
    gmsh.model.geo.addPlaneSurface([4], 4)

    gmsh.model.geo.addCurveLoop([-16, 6, 7, 17], 5)
    gmsh.model.geo.addPlaneSurface([5], 5)

    gmsh.model.geo.addCurveLoop([-21, 22, 9, 10], 6)
    gmsh.model.geo.addPlaneSurface([6], 6)

    gmsh.model.geo.addCurveLoop([-19, 20, 21, 11], 7)
    gmsh.model.geo.addPlaneSurface([7], 7)

    gmsh.model.geo.synchronize()

    ps = gmsh.model.addPhysicalGroup(2, [1,2,3,4,5,6,7],1)
    gmsh.model.setPhysicalName(1, ps, "My surface")

    gmsh.model.mesh.generate(2)

    gmsh.write("my_mesh.msh")
    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()

    gmsh.finalize()

    os.system('dolfin-convert my_mesh.msh my_mesh_n.xml')
    mesh = Mesh('my_mesh_n.xml')
    return mesh
#Plane electrode potential - approximation
def pe_p(u_0,aa,bb,cc):
    x, y, z, a, b, c = sp.symbols('x[0] x[1] z a b c')
    r = sp.sqrt(x**2+y**2+z**2)

    Lx = sp.atanh(x/r)
    Ly = sp.atanh(y/r)
    Lz = sp.atanh(z/r)

    Px = x*sp.atan(y*z/r/x)
    Py = y*sp.atan(x*z/r/y)
    Pz = z*sp.atan(y*x/r/z)

    F110 = y*Lx + x*Ly - Pz
    F101 = z*Lx + x*Lz - Py
    F011 = y*Lz + z*Ly - Px

    Is1 = F110.subs([(x,x-a),(y,y-b),(z,z-c)]) - F110.subs([(x,x+a),(y,y-b),(z,z-c)]) \
        - F110.subs([(x,x-a),(y,y+b),(z,z-c)]) + F110.subs([(x,x+a),(y,y+b),(z,z-c)])
    Is2 = F101.subs([(x,x-a),(y,y-b),(z,z-c)]) - F101.subs([(x,x+a),(y,y-b),(z,z-c)]) \
        - F101.subs([(x,x-a),(y,y-b),(z,z+c)]) + F101.subs([(x,x+a),(y,y-b),(z,z+c)])
    Is3 = F011.subs([(x,x-a),(y,y-b),(z,z-c)]) - F011.subs([(x,x-a),(y,y+b),(z,z-c)]) \
        - F011.subs([(x,x-a),(y,y-b),(z,z+c)]) + F011.subs([(x,x-a),(y,y+b),(z,z+c)])
    Is4 = Is1.subs([(c,-c)])
    Is5 = Is2.subs([(b,-b)])
    Is6 = Is3.subs([(a,-a)])

    u = (Is1 + Is2 + Is3 + Is4 + Is5 + Is6)/sp.pi/4
    u_n = u.subs([(a,aa), (b,bb), (c,cc), (z,-1.1*cc)])
    norm = u.subs([(a,aa), (b,bb), (c,cc), (z,-cc), (x,0), (y,0)])
    u_c = sp.ccode(u_0*u_n/norm)
    return u_c
#Plane electrode electric field - approximation
def pe_ef(aa,bb,cc):
    tol = 5
    x, y, z, a, b, c = sp.symbols('x[0] x[1] z a b c')
    r = sp.sqrt(x**2+y**2+z**2)

    Lx = sp.atanh(x/r)
    Ly = sp.atanh(y/r)
    Lz = sp.atanh(z/r)

    Px = x*sp.atan(y*z/r/x)
    Py = y*sp.atan(x*z/r/y)
    Pz = z*sp.atan(y*x/r/z)

    F110 = y*Lx + x*Ly - Pz
    F101 = z*Lx + x*Lz - Py
    F011 = y*Lz + z*Ly - Px

    Is1 = F110.subs([(x,x-a),(y,y-b),(z,z-c)]) - F110.subs([(x,x+a),(y,y-b),(z,z-c)]) \
        - F110.subs([(x,x-a),(y,y+b),(z,z-c)]) + F110.subs([(x,x+a),(y,y+b),(z,z-c)])
    Is2 = F101.subs([(x,x-a),(y,y-b),(z,z-c)]) - F101.subs([(x,x+a),(y,y-b),(z,z-c)]) \
        - F101.subs([(x,x-a),(y,y-b),(z,z+c)]) + F101.subs([(x,x+a),(y,y-b),(z,z+c)])
    Is3 = F011.subs([(x,x-a),(y,y-b),(z,z-c)]) - F011.subs([(x,x-a),(y,y+b),(z,z-c)]) \
        - F011.subs([(x,x-a),(y,y-b),(z,z+c)]) + F011.subs([(x,x-a),(y,y+b),(z,z+c)])
    Is4 = Is1.subs([(c,-c)])
    Is5 = Is2.subs([(b,-b)])
    Is6 = Is3.subs([(a,-a)])

    u = (Is1 + Is2 + Is3 + Is4 + Is5 + Is6)
    #u_n = u.subs([(a,aa), (b,bb), (c,cc)])
    u = sp.simplify(u)
    norm = u.subs([(a,aa), (b,bb), (c,cc), (z,-cc), (x,0), (y,0)])
    norm = norm.evalf(n=tol)
    E_x = -sp.diff(u/norm,x)
    E_x = E_x.evalf(n=tol)
    #E_x = sp.simplify(E_x)
    #E_x_n = E_x.subs([(a,aa), (b,bb), (c,cc), (z,-1.1*cc)])
    #E_x_n = sp.simplify(E_x_n)
    E_y = -sp.diff(u/norm,y)
    E_y = E_y.evalf(n=tol)
    #E_y = sp.simplify(E_y)
    #E_y_n = E_y.subs([(a,aa), (b,bb), (c,cc), (z,-1.1*cc)])
    #E_y_n = sp.simplify(E_y_n)
    E_z = -sp.diff(u/norm,z)
    E_z = E_z.evalf(n=tol)
    #E_z = sp.simplify(E_z)
    #E_z_n = E_z.subs([(a,aa), (b,bb), (c,cc), (z,-1.1*cc)])
    #E_z_n = sp.simplify(E_z_n)
    out = [sp.ccode(E_x), sp.ccode(E_y), sp.ccode(E_z)]
    return out

def rand_vec(m, mtp):
    array = m.vector().get_local()
    s = int(np.size(array))
    array = array + mtp*np.random.rand(s)
    m.vector()[:] = array
    return m

def SL_pot(u):
    FS = u.function_space()
    mesh = FS.mesh()
    
    d2v = dof_to_vertex_map(FS)
    
    from bempp.api.external import fenics 
    trace_space, trace_matrix = \
    fenics.fenics_to_bempp_trace_data(FS)
    
    func_fem_array = u.compute_vertex_values()
    func_bem_array = trace_matrix * func_fem_array[d2v]
    
    vertices = trace_space.grid.vertices
    x, y, z = vertices
    idx = z != np.max(z)
    func_bem_array[idx] = 0
    
    grid_func = bempp.api.GridFunction(trace_space, coefficients =func_bem_array)
    #bempp.api.PLOT_BACKEND = "paraview"
    #grid_func.plot()
    
#    bempp_space = bempp.api.function_space(trace_space.grid, "DP", 0)
#    print("FEM dofs: {0}".format(mesh.num_vertices()))
#    print("BEM dofs: {0}".format(bempp_space.global_dof_count))
#    print(trace_matrix.shape)
    
#    grid = bempp_space.grid
#    vertices = grid.vertices
    
    coord_T = np.transpose(mesh.coordinates())
    
    #@bempp.api.real_callable
    #def m_n(x, n, domain_index, result):
    #    #result[0] = np.sin(x[1])
    #    if x[2] == np.max(vertices[2]):
    #        result[0] = np.tanh(x[1])
    #    else:
    #        result[0] = 0
    
    from bempp.api.operators.potential import laplace as laplace_potential
    slp_pot = laplace_potential.single_layer(trace_space, coord_T)
    
    res = np.real(slp_pot.evaluate(grid_func))
    u2 = Function(FS)
    res_2 = res[:mesh.num_vertices()]
    u2.vector()[:] =   np.transpose(res_2[0][d2v])
    
    return u2

def SL_pot_s(u):
    FS = u.function_space()
    mesh = FS.mesh()
    
    d2v = dof_to_vertex_map(FS)
    
    from bempp.api.external import fenics 
    trace_space, trace_matrix = \
    fenics.fenics_to_bempp_trace_data(FS)
    
    func_fem_array = u.compute_vertex_values()
    func_bem_array = trace_matrix * func_fem_array[d2v]
    
    vertices = trace_space.grid.vertices
    x, y, z = vertices
    idx = z != np.max(z)
    func_bem_array[idx] = 0
    
    grid_func = bempp.api.GridFunction(trace_space, coefficients =func_bem_array)
    
    coord_T = np.transpose(mesh.coordinates())
    xx, yy, zz = coord_T
    idx2 = zz==np.max(zz)
    xx = xx[idx2]
    yy = yy[idx2]
    zz = zz[idx2]
    top_coord = np.array([xx, yy, zz])
    
    from bempp.api.operators.potential import laplace as laplace_potential
    slp_pot = laplace_potential.single_layer(trace_space, top_coord)
    res = np.real(slp_pot.evaluate(grid_func))
    res_2 = res[:mesh.num_vertices()]
    work_array  = res_2[0]
    final_array = np.array([])
    
    work_size = work_array.shape[0]
    array_size = func_fem_array.shape[0]
    N = int(array_size/work_size)
    if math.modf(N)[0] != 0:
        print("Mesh layers are different in z direction")
    
    for i in range(0,N,1):
       final_array = np.append(final_array, work_array)
        
    u2 = Function(FS)
    u2.vector()[:] = final_array[d2v] #np.transpose(final_array[d2v])
    return u2

#Hd = project(-grad(u2), W)
##Hd_x, Hd_y, Hd_z = Hd.split(deepcopy=True)
#vtkfile_hd_2 = File('hd_2.pvd')
#vtkfile_hd_2 << Hd

def pe_EF(a,b,c,Lx,Ly,angle,file_str):
    ## BEM part
    grid = bempp.api.shapes.cuboid(length=(2*a, 2*b, 2*c), origin=(-a, -b, c+0.5+a), h=3.0)
    coord = grid.vertices
    angle = math.pi/180*angle
    mat = np.array([[math.cos(angle), math.sin(angle), 0],
                     [-math.sin(angle), math.cos(angle), 0],
                     [0,0,1]])
    coord = mat.dot(coord)
    elems = grid.elements
    grid = bempp.api.Grid(coord,elems)
    space = bempp.api.function_space(grid, "DP", 0)

    @bempp.api.real_callable
    def one_fun(x, n, domain_index, res):
        res[0] = 1
        
    rhs = bempp.api.GridFunction(space, fun=one_fun)

    op = bempp.api.operators.boundary.laplace.single_layer(space, space, space)

    sol, _, iteration_count = bempp.api.linalg.gmres(op, rhs, tol = 1e-6, use_strong_form=True, return_iteration_count=True)
    
    print("Number of iterations: {0}".format(iteration_count))
    
    ## FEM part
    # Lx = 30
    # Ly = 15
    z_max = 0.5
    p1 = Point(-Lx/2,-Ly/2,-z_max) #matrix : cos(a)  sin(a)
    p2 = Point(Lx/2,Ly/2,z_max)    #         -sin(a) cos(a)
    nx = 570
    ny = 200
    mesh = BoxMesh(p1,p2,nx,ny,3)
    #coord_T = np.transpose(mesh.coordinates())
    
    El1 = FiniteElement('CG', tetrahedron, 2)
    FS = FunctionSpace(mesh, El1)
    
    coord_T = FS.tabulate_dof_coordinates().T
    
    from bempp.api.operators.potential import laplace as laplace_potential
    slp_pot = laplace_potential.single_layer(space, coord_T)
    res = np.real(slp_pot.evaluate(sol))
    
    u = Function(FS)
    u.vector()[:] = res[0]
    #vtkfile_m = File('pot.pvd')
    
    El_1 = FiniteElement('CG', tetrahedron, 1)
    FS_1 = FunctionSpace(mesh, El_1)
    
    d2v = dof_to_vertex_map(FS_1)
    #v2d = vertex_to_dof_map(FS_1)
    
    E1 = project(-grad(u)[0],FS_1)
    E2 = project(-grad(u)[1],FS_1)
    E3 = project(-grad(u)[2],FS_1)
    
    dE1dz = project(E1.dx(2),FS_1)
    dE2dz = project(E2.dx(2),FS_1)
    dE3dz = project(E3.dx(2),FS_1)
    
    # vtk_E1 = File('/media/mnv/A2E41E9EE41E74AF/graphs/E/E1.pvd')
    # vtk_E2 = File('/media/mnv/A2E41E9EE41E74AF/graphs/E/E2.pvd')
    # vtk_E3 = File('/media/mnv/A2E41E9EE41E74AF/graphs/E/E3.pvd')
    # vtk_E1 << E1
    # vtk_E2 << E2
    # vtk_E3 << E3
    
    ## FEM-BEM part
    from bempp.api.external import fenics as fen
    trace_space, trace_matrix = \
    fen.fenics_to_bempp_trace_data(FS_1)

    E1_array = E1.compute_vertex_values()
    E2_array = E2.compute_vertex_values()
    E3_array = E3.compute_vertex_values()
    
    
    E1_array = trace_matrix * E1_array[d2v]
    E2_array = trace_matrix * E2_array[d2v]
    E3_array = trace_matrix * E3_array[d2v]
    
    dE1dz_array = dE1dz.compute_vertex_values()
    dE2dz_array = dE2dz.compute_vertex_values()
    dE3dz_array = dE3dz.compute_vertex_values()
    
    dE1dz_array = trace_matrix * dE1dz_array[d2v]
    dE2dz_array = trace_matrix * dE2dz_array[d2v]
    dE3dz_array = trace_matrix * dE3dz_array[d2v]
    
    vertices = trace_space.grid.vertices
    elems = trace_space.grid.elements

    x, y, z = vertices
    idx = z == np.max(z)

    elems_x, elems_y, elems_z = elems

    x = x[idx]
    y = y[idx]
    z = z[idx]

    Vert = np.array([x,y]) ## [x,y,z]
    top_elems_ind = np.array([],dtype = int)
    top_elems_X = np.array([],dtype = int)
    top_elems_Y = np.array([],dtype = int)
    top_elems_Z = np.array([],dtype = int)

    for i in range(trace_space.grid.number_of_elements):
        j = 0
        for a in vertices[:, elems[:, i]][2]:
            if a == z_max:
                j += 1
        if j == 3:
            top_elems_ind = np.append(top_elems_ind,i)
            top_elems_X = np.append(top_elems_X,elems_x[i])
            top_elems_Y = np.append(top_elems_Y,elems_y[i])
            top_elems_Z = np.append(top_elems_Z,elems_z[i])
            

    top_elems = np.array([top_elems_X,top_elems_Y,top_elems_Z])
    top_elems = top_elems - np.min(top_elems)
    
    route_0 = '/home/mnv/llg_nl/'
    
    MESH = meshio.Mesh(Vert.T,[("triangle",top_elems.T)])
    MESH.write(route_0 + 'MESH.xml')
    mesh_2 = Mesh(route_0 + 'MESH.xml')
    El2 = FiniteElement('CG', triangle, 1)
    FS2 = FunctionSpace(mesh_2, El2)
    
    E1 = E1_array[idx]
    E2 = E2_array[idx]
    E3 = E3_array[idx]
    
    dE1dz = dE1dz_array[idx]
    dE2dz = dE2dz_array[idx]
    dE3dz = dE3dz_array[idx]
    
    e1 = Function(FS2)
    e2 = Function(FS2)
    e3 = Function(FS2)
    
    dE1dz_f = Function(FS2)
    dE2dz_f = Function(FS2)
    dE3dz_f = Function(FS2)
    
    d2v = dof_to_vertex_map(FS2)
    #v2d = vertex_to_dof_map(FS2)
    
    e1.vector()[:] = E1[d2v]
    e2.vector()[:] = E2[d2v]
    e3.vector()[:] = E3[d2v]
    
    dE1dz_f.vector()[:] = dE1dz[d2v]
    dE2dz_f.vector()[:] = dE2dz[d2v]
    dE3dz_f.vector()[:] = dE3dz[d2v]
    
    El_3 = VectorElement('CG', triangle, 1, dim=3)
    FS_3 = FunctionSpace(mesh_2, El_3)
    
    e_v = project(as_vector((e1,e2,e3)), FS_3)
    dEdz_v = project(as_vector((dE1dz_f, dE2dz_f, dE3dz_f)), FS_3)
    
    E_series = TimeSeries(file_str+'E_mid_20')
    dEdz_series = TimeSeries(file_str+'E_mid_20_dEdz')
    
    E_series.store(e_v.vector(),0)
    dEdz_series.store(dEdz_v.vector(),0)
    
    hdf_E = HDF5File(mesh_2.mpi_comm(), file_str + "E_hdf_20.h5", 'w')
    hdf_E.write(mesh_2, "/my_mesh")
    hdf_E.write(e_v, "/e_field")
    hdf_E.write(dEdz_v, "/dedz_field")
    hdf_E.close()
    
    gc.collect()
    return 0 #[FS2, FS, FS_1, FS_3, e_v]

def s_chg(m3, SL_space, FS_2, FS_3, FS_3_1, FS_2v, idx, space_top, slp_pot, trace_space, trace_matrix):
    m3 = project(m3,SL_space)
    # mesh_2 = SL_space.mesh()
    # coord = mesh_2.coordinates().T ## можно не обновлять
    # x, y = coord
    # tol = 1e-14
    
    # idx1 = (x != np.max(x))
    # idx2 = (x != np.min(x))
    # idx3 = (y != np.max(y))
    # idx4 = (y != np.min(y))
    # idx = idx1 & idx2 & idx3 & idx4
    # z = np.zeros(x.shape) + 1
    # coord = np.array([x,y,z]) ## можно не обновлять
    # elems = mesh_2.cells().T ## можно не обновлять
    
    # grid_top = bempp.api.Grid(coord, elems) ## не обновлять!
    # space_top = bempp.api.function_space(grid_top, "P", 1) ## не обновлять!
    
    func_array_top = m3.compute_vertex_values() #m3.compute_vertex_values() m3.vector().get_local()
    d2v = dof_to_vertex_map(SL_space)
    
    grid_func_top = bempp.api.GridFunction(space_top, coefficients = 2*func_array_top[idx])
    #grid_func_top.plot()
    
    # @bempp.api.real_callable
    # def m_n(x, n, domain_index, res):
    #     res[0] = np.tanh(x[1])

    # grid_fun = bempp.api.GridFunction(space_top, fun=m_n)
    
    #mesh_3 = FS_3_1.mesh()
    #coord = FS_3.tabulate_dof_coordinates().T ## можно не обновлять
    
    # from bempp.api.external import fenics as fen
    # trace_space, trace_matrix = fen.fenics_to_bempp_trace_data(FS_3_1) ## не обновлять!
    
    # from bempp.api.operators.potential import laplace as laplace_potential
    # slp_pot = laplace_potential.single_layer(space_top, coord)
    res = slp_pot.evaluate(grid_func_top)  ## grid_func_top ### HUGE
    
    u = Function(FS_3)
    u.vector()[:] = res[0]  ## вычислен потенциал от "верхних" магнитных зарядов
    
    H1 = project(-grad(u)[0],FS_3_1)
    H2 = project(-grad(u)[1],FS_3_1)
    H3 = project(-grad(u)[2],FS_3_1)
    
    d2v = dof_to_vertex_map(FS_3_1)

    H1_array = H1.compute_vertex_values() #.compute_vertex_values()
    H2_array = H2.compute_vertex_values() #.compute_vertex_values()
    H3_array = H3.compute_vertex_values() #.compute_vertex_values()
    
    H1_array = trace_matrix * H1_array[d2v]
    H2_array = trace_matrix * H2_array[d2v]
    H3_array = trace_matrix * H3_array[d2v]
    
    vertices = trace_space.grid.vertices
    x, y, z = vertices
    idx = z == np.max(z)
    
    H1 = H1_array[idx]
    H2 = H2_array[idx]
    H3 = H3_array[idx]
    
    h1 = Function(FS_2)
    h2 = Function(FS_2)
    h3 = Function(FS_2)
    
    d2v = dof_to_vertex_map(FS_2)
    
    h1.vector()[:] = H1[d2v]
    h2.vector()[:] = H2[d2v]
    h3.vector()[:] = H3[d2v]
    
    hd_s = project(as_vector((h1,h2,h3)), FS_2v)
    return hd_s

def s_chg_prep(SL_space, FS_2, FS_3, FS_3_1, FS_2v, z_pl): #SL_space, FS_2, FS_3, FS_3_1, FS_2v
    
    mesh_2 = SL_space.mesh()
    coord = mesh_2.coordinates().T ## можно не обновлять
    x, y = coord
    #tol = 1e-14
    
    idx1 = (x != np.max(x))
    idx2 = (x != np.min(x))
    idx3 = (y != np.max(y))
    idx4 = (y != np.min(y))
    idx = idx1 & idx2 & idx3 & idx4
    z = np.zeros(x.shape) + z_pl
    coord = np.array([x,y,z]) ## можно не обновлять
    elems = mesh_2.cells().T ## можно не обновлять
    
    grid_top = bempp.api.Grid(coord, elems) ## не обновлять!
    space_top = bempp.api.function_space(grid_top, "P", 1) ## не обновлять!
    
    from bempp.api.external import fenics as fen
    trace_space, trace_matrix = fen.fenics_to_bempp_trace_data(FS_3_1)
    
    coord = FS_3.tabulate_dof_coordinates().T
    
    from bempp.api.operators.potential import laplace as laplace_potential
    slp_pot = laplace_potential.single_layer(space_top, coord)
    
    return [idx, space_top, slp_pot, trace_space, trace_matrix]

def phi_t(y, z, Ly, Z):
    ly, z1 = sp.symbols('ly z1')
    z_m = z-z1
    z_p = z+z1
    y_m = y-ly
    y_p = y+ly
    
    phi = 2*z_m*(sp.atan(y_m/z_m) - sp.atan(y_p/z_m)) \
        + 2*z_p*(sp.atan(y_p/z_p) - sp.atan(y_m/z_p)) \
             + y_m*sp.log((y_m**2 + z_m**2)/(y_m**2 + z_p**2)) \
                 - y_p*sp.log((y_p**2 + z_m**2)/(y_p**2 + z_p**2))
                 
    #phi = y_m*sp.ln((y_m**2 + z_m**2)/(y_m**2 + z_p**2)) #- y_m*sp.ln((y_p**2 + z_m**2)/(y_p**2 + z_p**2))
    phi = phi.subs([(ly,Ly),(z1,Z)])
    return phi

def pair(y, dy, l, z, Z):
    phi = phi_t(y-dy,z,l,Z) - phi_t(y+dy,z,l,Z)
    return phi

def n_pair(Ly1,l1,Z1,Z01,n):
    y, z = sp.symbols('x[1] z')
    ly, z1 = sp.symbols('ly z1')
    Ly,l,Z,Z0 = sp.symbols('Ly l Z Z0')
    z_m = z-z1
    z_p = z+z1
    y_m = y-ly
    y_p = y+ly
    
    phi = 2*z_m*(sp.atan(y_m/z_m) - sp.atan(y_p/z_m)) \
        + 2*z_p*(sp.atan(y_p/z_p) - sp.atan(y_m/z_p)) \
             + y_m*sp.ln((y_m**2 + z_m**2)/(y_m**2 + z_p**2)) \
                 - y_p*sp.ln((y_p**2 + z_m**2)/(y_p**2 + z_p**2)) ## y, z, ly, z1
    
    dy = l/2+Ly/4
    phi = phi.subs([(y,y-dy), (ly,dy), (z1,Z)]) - phi.subs([(y,y+dy), (ly,dy), (z1,Z)])
    ## y,z,l,Ly,Z
    dy = 3*l/2
    i = 1
    while i < n:
        phi = phi + (-1)**i*phi.subs([(y,y-dy), (ly,l/2), (z1,Z)]) - phi.subs([(y,y+dy), (ly,l/2), (z1,Z)])
        dy = dy+l
        i += 1
    
    #phi_n = phi.subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)])
    #print(phi_n)
    #sp.plot(phi_n,(y,-2*n*l1,2*n*l1))
    
    hy = -sp.diff(phi,y).subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)])
    hz = -sp.diff(phi,z).subs([(l,l1), (z,Z01), (Z,Z1), (Ly,Ly1)])
    #sp.plot(hz,(y,-2*n*l1,2*n*l1))
    
    hy_c = sp.ccode(hy)
    hz_c = sp.ccode(hz)
    print(hy_c)
    llog = sp.ln(y)
    llog = sp.printing.ccode(llog)
    
    out = Expression(llog, degree = 3, l = l1, Ly = Ly1, Z = Z1, z = Z01)#, degree = 3, z=Z-1
    return out

def side_pot(FS_2, FS_3, FS_3_1, FS_2v, z_top, z_max, Lx, Ly, l_i):
    nx = int(1*l_i)
    ny = int(2*Ly)
    mesh_top = RectangleMesh(Point(-(Lx/2+l_i), -Ly/2),Point(-Lx/2,Ly/2), nx, ny)
    
    coord = mesh_top.coordinates().T
    x,y = coord
    z = np.zeros(x.shape) + z_top + z_max
    coord = [x,y,z]
    elems = mesh_top.cells().T
    grid_top = bempp.api.Grid(coord, elems)
    space_top = bempp.api.function_space(grid_top, "P", 1, include_boundary_dofs=True)
    
    coord = FS_3.tabulate_dof_coordinates().T
    
    from bempp.api.operators.potential import laplace as laplace_potential
    slp_pot = laplace_potential.single_layer(space_top, coord)
    
    @bempp.api.real_callable
    def m_n(x, n, domain_index, res):
        res[0] = np.tanh(x[1])
    
    grid_fun = bempp.api.GridFunction(space_top, fun = m_n)
    res = slp_pot.evaluate(grid_fun)
    
    from bempp.api.external import fenics as fen
    trace_space, trace_matrix = fen.fenics_to_bempp_trace_data(FS_3_1)
    
    u = Function(FS_3)
    u.vector()[:] = res[0]  ## вычислен потенциал от "верхних" магнитных зарядов
    
    H1 = project(-grad(u)[0],FS_3_1)
    H2 = project(-grad(u)[1],FS_3_1)
    H3 = project(-grad(u)[2],FS_3_1)
    
    d2v = dof_to_vertex_map(FS_3_1)

    H1_array = H1.compute_vertex_values() #.compute_vertex_values()
    H2_array = H2.compute_vertex_values() #.compute_vertex_values()
    H3_array = H3.compute_vertex_values() #.compute_vertex_values()
    
    H1_array = trace_matrix * H1_array[d2v]
    H2_array = trace_matrix * H2_array[d2v]
    H3_array = trace_matrix * H3_array[d2v]
    
    vertices = trace_space.grid.vertices
    x, y, z = vertices
    idx = z == np.max(z)
    
    H1 = H1_array[idx]
    H2 = H2_array[idx]
    H3 = H3_array[idx]
    
    h1 = Function(FS_2)
    h2 = Function(FS_2)
    h3 = Function(FS_2)
    
    d2v = dof_to_vertex_map(FS_2)
    
    h1.vector()[:] = H1[d2v]
    h2.vector()[:] = H2[d2v]
    h3.vector()[:] = H3[d2v]
    
    hd_s = project(as_vector((h1,h2,h3)), FS_2v)
    
    return hd_s

def aa_min(Nu, Np, Mat_1, Ku, Kp, Kc):
    
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
    
    nx = 100
    ny = 200
    
    theta = np.linspace(0, np.pi, nx)
    phi = np.linspace(0, 2*np.pi, ny)
    test = np.zeros((nx,ny))
    
    print('before test')
    
    for i in range(0, nx, 1):
        for j in range(0, ny, 1):
            test[i,j] = w(theta[i], phi[j], Nu, Np, Mat_1, Ku, Kp, Kc)
    
    bounds = [(np.pi/2, np.pi), (0*np.pi, 2*np.pi)]
    #from scipy import optimize
    print('before annealing')
    results = scipy.optimize.dual_annealing(wm, bounds, args=(Nu, Np, Mat_1, Ku, Kp, Kc))

    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #cont = ax.contourf(phi, theta, test, levels = 25)
    #ax.plot(results.x[1],results.x[0], marker ='o', color = 'Red')
    #ax.set_xlabel('$\phi$')
    #ax.set_ylabel('$\Theta$')
    #plt.colorbar(cont)
    #plt.show()
    #fig.savefig('s30_min.png', dpi = 300)
    return results

def pe_analitic():
    """
    Function returns C-expression (in analytical form) for electric field and its derivatives along z; 
    Electrod length = L, rotation angle  = phi
    """
    L, x, y, z, phi = sp.symbols('L x[0] x[1] z phi')
    
    x_rot = sp.cos(phi)*x - sp.sin(phi)*y
    
    u_0 = -1/sp.pi*(sp.atan((L/2-x_rot)/z) - sp.atan((-L/2-x_rot)/z))
    
    e_x = -sp.simplify(sp.diff(u_0, x))
    e_y = -sp.simplify(sp.diff(u_0, y))
    e_z = -sp.simplify(sp.diff(u_0, z))
    
    dex_dz = -sp.simplify(sp.diff(e_x,z))
    dey_dz = -sp.simplify(sp.diff(e_y,z))
    dez_dz = -sp.simplify(sp.diff(e_z,z))
    
    e_x_c = sp.ccode(e_x)
    e_y_c = sp.ccode(e_y)
    e_z_c = sp.ccode(e_z)
    
    dex_dz_c = sp.ccode(dex_dz)
    dey_dz_c = sp.ccode(dey_dz)
    dez_dz_c = sp.ccode(dez_dz)
    
    return [e_x_c, e_y_c, e_z_c, dex_dz_c, dey_dz_c, dez_dz_c]
