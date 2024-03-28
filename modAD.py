import fenics as fe
import dolfin as df
import time
import numpy as np
from dolfin import LagrangeInterpolator, refine, MeshFunction, split, grad, MPI
from fenics import project, MixedElement, FunctionSpace, TestFunctions, Function, derivative
from fenics import NonlinearVariationalProblem, NonlinearVariationalSolver

def Value_Coor_dof(solution_vector_pf, spaces_pf, comm):
    """Return value of the solution at the degrees of freedom and corresponding coordinates."""


    v_phi = spaces_pf[0]
    (Phi_answer, U_answer) = split(solution_vector_pf)
    coordinates_of_all = v_phi.tabulate_dof_coordinates()
    grad_Phi = project(fe.sqrt(fe.dot(grad(Phi_answer), grad(Phi_answer))), v_phi)
    phi_value_on_dof = grad_Phi.vector().get_local()

    all_Val_dof = comm.gather(phi_value_on_dof, root=0)
    all_point = comm.gather(coordinates_of_all, root=0)

    # Broadcast the data to all processors
    all_point = comm.bcast(all_point, root=0)
    all_Val_dof = comm.bcast(all_Val_dof, root=0)

    # Combine the data from all processors
    all_Val_dof_1 = [val for sublist in all_Val_dof for val in sublist]
    all_point_1 = [point for sublist in all_point for point in sublist]

    point = np.array(all_point_1)
    Val_dof = np.array(all_Val_dof_1)

    return Val_dof, point

def Coordinates_Of_Int(interface_threshold_gradient, solution_vector_pf, spaces_pf, comm):
    """Get the small mesh and return coordinates of the interface."""
    dof_Val, dof_Coor = Value_Coor_dof(solution_vector_pf, spaces_pf, comm)

    high_gradient_indices = np.where(dof_Val > interface_threshold_gradient)[0]
    Coord_L_Of_Int = dof_Coor[high_gradient_indices]


    return Coord_L_Of_Int


def mark_coarse_mesh(mesh_coarse, list_coordinate_points_interface):
    """Mark the cells in the coarse mesh that have the interface points in them so they can be refined."""
    mf = MeshFunction("bool", mesh_coarse, mesh_coarse.topology().dim(), False)
    len_mf = len(mf)
    Cell_Id_List = []

    tree = mesh_coarse.bounding_box_tree()

    for Cr in list_coordinate_points_interface:
        cell_id = tree.compute_first_entity_collision(df.Point(Cr))
        if cell_id != 4294967295 and 0 <= cell_id < len_mf:
            Cell_Id_List.append(cell_id)

    Cell_Id_List = np.unique(np.array(Cell_Id_List, dtype=int))
    mf.array()[Cell_Id_List] = True

    return mf


def refine_to_min(mesh_coarse, list_coordinate_points_interface):
    """Refine coarse mesh cells that contain the interface coordinate."""

    mf = mark_coarse_mesh(mesh_coarse, list_coordinate_points_interface)
    mesh_new = fe.refine(mesh_coarse, mf, redistribute=True)

    return mesh_new


def refine_mesh(physical_parameters_dict, coarse_mesh, solution_vector_pf, spaces_pf, comm ):
    """Refines the mesh based on provided parameters and updates related variables and solvers."""
    
    max_level = physical_parameters_dict['max_level']
    interface_threshold_gradient = physical_parameters_dict["interface_threshold_gradient"]

    coarse_mesh_it = coarse_mesh

    # Get the coordinates of the points that wants to be refined
    list_coordinate_points_interface = Coordinates_Of_Int(interface_threshold_gradient, solution_vector_pf, spaces_pf, comm)

    # Refine the mesh up to the maximum level specified
    for res in range(max_level):
        mesh_new = refine_to_min(coarse_mesh_it, list_coordinate_points_interface)
        coarse_mesh_it = mesh_new

    # mesh information:
    mesh_info = {
        'n_cells': df.MPI.sum(comm, mesh_new.num_cells()),
        'hmin': df.MPI.min(comm, mesh_new.hmin()),
        'hmax': df.MPI.max(comm, mesh_new.hmax()),
        'dx_min': df.MPI.min(comm, mesh_new.hmin()) / df.sqrt(2),
        'dx_max': df.MPI.max(comm, mesh_new.hmax()) / df.sqrt(2),
    }


    return mesh_new, mesh_info

