import bempp.api
import numpy as np
import math
from fenics import *
def pe_EF(a,b,c,Lx,Ly,Lz,angle):
    ## BEM part
    grid = bempp.api.shapes.cuboid(length=(2*a, 2*b, 2*c), origin=(-a, -b, -c), h=3)
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
    bempp.api.export('/home/mnv/llg_nl/electric_field_clip/surf_func.vtu', grid_function=sol)
    
    print("Number of iterations: {0}".format(iteration_count))
    
    ## FEM part
    # Lx = 30
    # Ly = 15
    #z_max = 0.5
    p1 = Point(-Lx/2,-Ly/2,-Lz) #matrix : cos(a)  sin(a)
    p2 = Point(Lx/2,Ly/2,0)    #         -sin(a) cos(a)
    nx = 100
    ny = 10
    nz = 60
    mesh = BoxMesh(p1,p2,nx,ny,nz)
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
    
    #El_3 = VectorElement('CG', tetrahedron, 1, dim=3)
    #FS_3 = FunctionSpace(mesh, El_3)
    
    #e_v = project(as_vector((E1,E2,E3)), FS_3)
    
    e1_file =  XDMFFile('/home/mnv/llg_nl/electric_field_clip/e1.xdmf')
    e1_file.write(E1)
    e1_file.close()

    e2_file =  XDMFFile('/home/mnv/llg_nl/electric_field_clip/e2.xdmf')
    e2_file.write(E2)
    e2_file.close()

    e3_file =  XDMFFile('/home/mnv/llg_nl/electric_field_clip/e3.xdmf')
    e3_file.write(E3)
    e3_file.close()
    
    return 0 #[FS2, FS, FS_1, FS_3, e_v]
a = 90
b = 200
c = 1
Lx = 200
Ly = 20
Lz = 180
angle = 0
pe_EF(a,b,c,Lx,Ly,Lz,angle)
