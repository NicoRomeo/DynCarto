"""
Simulation of incompressible 2D channel flow past a circular obstacle. 
Integrates the Navier-Stokes equation using FeniCS, adapted directly from the code by Jorgen S. Dokken
https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html

"""

import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.interpolate as scip
import h5py

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element

from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities
from dolfinx import plot
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)

from tqdm import tqdm

gmsh.initialize()


def run_simulation(mu_in, Trun, num_saves, T_steady):
    """Runs simulation of flow past a channel at viscosity mu"""
    gmsh.initialize()
    ###### Mesh construction
    L = 2.2
    H = 0.41
    c_x = c_y = 0.2
    r = 0.05
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
        obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

    if mesh_comm.rank == model_rank:
        fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
        gmsh.model.occ.synchronize()

    fluid_marker = 1
    if mesh_comm.rank == model_rank:
        volumes = gmsh.model.getEntities(dim=gdim)
        assert (len(volumes) == 1)
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
        gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

    inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
    inflow, outflow, walls, obstacle = [], [], [], []
    if mesh_comm.rank == model_rank:
        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H / 2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H / 2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])
        gmsh.model.addPhysicalGroup(1, walls, wall_marker)
        gmsh.model.setPhysicalName(1, wall_marker, "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
        gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
        gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
        gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

    # Create distance field from obstacle.
    # Add threshold of mesh sizes based on the distance field
    # LcMax -                  /--------
    #                      /
    # LcMin -o---------/
    #        |         |       |
    #       Point    DistMin DistMax
    res_min = r / 3
    if mesh_comm.rank == model_rank:
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize("Netgen")

    #### load mesh 

    mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
    ft.name = "Facet markers"
    t = 0
    T = Trun                     # Final time
    dt = 1 / 1600                 # Time step size
    num_steps = int(T / dt)

    num_checks = int(num_steps / num_saves)
    k = Constant(mesh, PETSc.ScalarType(dt))
    mu = Constant(mesh, PETSc.ScalarType(mu_in))  # Dynamic viscosity  original value 0.001
    rho = Constant(mesh, PETSc.ScalarType(1))     # Density
    
    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)

    fdim = mesh.topology.dim - 1

    # Define boundary conditions


    class InletVelocity():
        def __init__(self, t):
            self.t = t

        def __call__(self, x):
            values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
            if self.t > T_steady:
                values[0] =  4* 1.5 * x[1] * (0.41 - x[1]) / (0.41**2)
            else:
                values[0] = 4 * 1.5 * np.sin(self.t * np.pi / (2*T_steady)) * x[1] * (0.41 - x[1]) / (0.41**2)
            return values


    # Inlet
    u_inlet = Function(V)
    inlet_velocity = InletVelocity(t)
    u_inlet.interpolate(inlet_velocity)
    bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
    # Walls
    u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bcu_walls = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
    # Obstacle
    bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
    bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
    # Outlet
    bcp_outlet = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
    bcp = [bcp_outlet]
    
    # functions space 
    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)
    u_.name = "u"
    u_s = Function(V)
    u_n = Function(V)
    u_n1 = Function(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)
    p_ = Function(Q)
    p_.name = "p"
    phi = Function(Q)
    # assemble matrices
    
    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    F1 = rho / k * dot(u - u_n, v) * dx
    F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
    F1 += 0.5 * mu * inner(grad(u + u_n), grad(v)) * dx - dot(p_, div(v)) * dx
    F1 += dot(f, v) * dx
    a1 = form(lhs(F1))
    L1 = form(rhs(F1))
    A1 = create_matrix(a1)
    b1 = create_vector(L1)
    a2 = form(dot(grad(p), grad(q)) * dx)
    L2 = form(-rho / k * dot(div(u_s), q) * dx)
    A2 = assemble_matrix(a2, bcs=bcp)
    A2.assemble()
    b2 = create_vector(L2)

    a3 = form(rho * dot(u, v) * dx)
    L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
    A3 = assemble_matrix(a3)
    A3.assemble()
    b3 = create_vector(L3)
    
    # Solver for step 1
    solver1 = PETSc.KSP().create(mesh.comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    pc1 = solver1.getPC()
    pc1.setType(PETSc.PC.Type.JACOBI)

    # Solver for step 2
    solver2 = PETSc.KSP().create(mesh.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.MINRES)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    pc2.setHYPREType("boomeramg")

    # Solver for step 3
    solver3 = PETSc.KSP().create(mesh.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    pc3 = solver3.getPC()
    pc3.setType(PETSc.PC.Type.SOR)
    n = -FacetNormal(mesh)  # Normal pointing out of obstacle
    dObs = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker)
    u_t = inner(as_vector((n[1], -n[0])), u_)
    drag = form(2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dObs)
    lift = form(-2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dObs)
    if mesh.comm.rank == 0:
        C_D = np.zeros(num_steps, dtype=PETSc.ScalarType)
        C_L = np.zeros(num_steps, dtype=PETSc.ScalarType)
        t_u = np.zeros(num_steps, dtype=np.float64)
        t_p = np.zeros(num_steps, dtype=np.float64)
    tree = bb_tree(mesh, mesh.geometry.dim)
    points = np.array([[0.15, 0.2, 0], [0.25, 0.2, 0]])
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, points)
    front_cells = colliding_cells.links(0)
    back_cells = colliding_cells.links(1)
    if mesh.comm.rank == 0:
        p_diff = np.zeros(num_steps, dtype=PETSc.ScalarType)
        
    # solving the time-dependent problem 

    saved_steps = []
    for i in tqdm(range(num_steps)):
        # Update current time step
        t += dt
        # Update inlet velocity
        inlet_velocity.t = t
        u_inlet.interpolate(inlet_velocity)

        # Step 1: Tentative velocity step
        A1.zeroEntries()
        assemble_matrix(A1, a1, bcs=bcu)
        A1.assemble()
        with b1.localForm() as loc:
            loc.set(0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu)
        solver1.solve(b1, u_s.x.petsc_vec)
        u_s.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with b2.localForm() as loc:
            loc.set(0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp)
        solver2.solve(b2, phi.x.petsc_vec)
        phi.x.scatter_forward()

        p_.x.petsc_vec.axpy(1, phi.x.petsc_vec)
        p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with b3.localForm() as loc:
            loc.set(0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.x.petsc_vec)
        u_.x.scatter_forward()

        # Update variable with solution form this time step
        with u_.x.petsc_vec.localForm() as loc_, u_n.x.petsc_vec.localForm() as loc_n, u_n1.x.petsc_vec.localForm() as loc_n1:
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)

        # Compute physical quantities
        # For this to work in paralell, we gather contributions from all processors
        # to processor zero and sum the contributions.
        drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
        lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)
        p_front = None
        if len(front_cells) > 0:
            p_front = p_.eval(points[0], front_cells[:1])
        p_front = mesh.comm.gather(p_front, root=0)
        p_back = None
        if len(back_cells) > 0:
            p_back = p_.eval(points[1], back_cells[:1])
        p_back = mesh.comm.gather(p_back, root=0)
        if mesh.comm.rank == 0:
            t_u[i] = t
            t_p[i] = t - dt / 2
            C_D[i] = sum(drag_coeff)
            C_L[i] = sum(lift_coeff)
            # Choose first pressure that is found from the different processors
            for pressure in p_front:
                if pressure is not None:
                    p_diff[i] = pressure[0]
                    break
            for pressure in p_back:
                if pressure is not None:
                    p_diff[i] -= pressure[0]
                    break

        # Downsample the data and save time step data
        if i % num_checks ==0:
            #with u_n.x.petsc_vec.localForm() as loc_:
            saved_steps.append(np.copy(u_.x.array.real[:]))
    print("---Integration done---\n")
    topology, cell_types, geometry = plot.vtk_mesh(V)
    gmsh.finalize()
    return saved_steps, geometry


def make_movie(filename, data, geometry):
    
    L = 2.2
    H = 0.41
    c_x = c_y = 0.2
    r = 0.05
    y = np.linspace(0, H, 160)
    x = np.linspace(0, L, 350)
    
    xx, yy = np.meshgrid(x,y)
    xi = np.array([[x0, y0] for x0 in x for y0 in y])
    
    r_cyl = (0.41 - 0.16 - 0.15)/2
    xcent = 0.152 + r_cyl
    ycent = 0.155 + r_cyl
    mask = (xx.T- xcent) **2 + (yy.T - ycent)**2 < r**2

    theta = np.linspace(0, 2*np.pi, 50)
    xc = r_cyl * np.cos(theta) + xcent
    yc = r_cyl * np.sin(theta) + ycent
     
    fps = 30
    fig = plt.figure()

    values = np.zeros((geometry.shape[0], 2), dtype=np.float64)
    u0 = data[0]
    values[:, :] = u0.reshape((geometry.shape[0], 2))
    vp = scip.griddata(geometry[:,:2], values[:,0], xi)

    mat = vp.reshape(len(x), len(y))
    # mask out the middle
    mat[mask] = np.nan
    # color bounds
    max_speed = np.max([np.max(u) for u in data])
    min_speed = np.min([np.min(u) for u in data])

    im = plt.imshow(mat.T, origin='upper', extent=(x[0], x[-1], y[0], y[-1]), cmap=plt.cm.Blues, vmax=max_speed, vmin=min_speed)
    plt.plot(xc, yc, color='k', lw=2.5)
    
    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        ui = data[i]
        values[:, :] = ui.reshape((geometry.shape[0], 2))
        vpi = scip.griddata(geometry[:,:2], values[:,0], xi)

        mat = vpi.reshape(len(x), len(y))
        mat[mask] = np.nan
        im.set_array(mat.T)
        return [im]

    anim = animation.FuncAnimation(
                                   fig, 
                                   animate_func, 
                                   blit=True,
                                   frames = len(data),
                                   interval = 1000 / fps, # in ms
                                   )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])
    return None



    



if __name__ == "__main__":
    ## uncomment for a test run
    #print("Test run....")
    mu_in = 0.001
    Trun = 8
    num_saves = 450
    T_steady = 0.2
    #data, geom = run_simulation(mu_in, Trun, num_saves, T_steady)
    #make_movie('test_vortex.mp4', data, geom)
    #print("~ test complete ~")
    
    #mu_arr = np.linspace(0.0007, 0.0025, 20)
    ### batch of simulations
    mu_arr = np.linspace(0.0026, 0.005, 10)
    for i, mu in enumerate(mu_arr):
        print("++++ i = {0}".format(i))
        data, geom = run_simulation(mu, Trun, num_saves, T_steady)
        make_movie('./vortexflow/vortex_mu{0}.mp4'.format(mu), data, geom)
        np.savez('./vortexflow/vortex_mu{0}.npz'.format(mu), *data)
    print("~sucess~")
    
    