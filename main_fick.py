from pathlib import Path

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as onp
import scipy.integrate as ode

import FEMcommon.load_mesh as mload
from FEMcommon.local_matrices import local_stiffness, local_vector, local_mass
from FEMcommon.assemble_global import assemble_global_matrix, assemble_global_vector
import FEMcommon.aux_helpers as helpers


def calculate_stiffness_matrix(mesh: mload.Mesh):
    gstiffness = assemble_global_matrix(mesh, local_stiffness)
    return jnp.array(gstiffness)


def calculate_mass_matrix(mesh: mload.Mesh):
    gmass = assemble_global_matrix(mesh, local_mass)
    gmass = helpers.concentrate_mass_matrix(gmass)
    return jnp.array(gmass)


def initial_cond(mesh: mload.Mesh):
    nodes = mesh.nodes
    inital_temp = 70
    hot_radius = 0.4

    return jnp.where(jnp.linalg.norm(nodes, ord=2, axis=1) <= hot_radius, inital_temp, 0)


def solve(Tmax, mesh: mload.Mesh):
    q0 = initial_cond(mesh)
    mass = calculate_mass_matrix(mesh)
    stiffness = calculate_stiffness_matrix(mesh)
    system_mtx = - jnp.linalg.inv(mass) @ stiffness

    # # move to the cpu
    # q0 = onp.array(q0)
    # system_mtx = onp.array(system_mtx)
    print("Matrix calculations done, moving onto solving ODE")

    def rhs(t, q):
        return system_mtx.dot(q)

    solution = ode.solve_ivp(rhs, [0, Tmax], q0)
    print("ODE solution done")
    return solution


def plot(mesh, values):
    xs = mesh.nodes[:, 0]
    ys = mesh.nodes[:, 1]
    zs = values
    ax = plt.axes(projection='3d')
    ax.set_zlim(0,100)
    ax.plot_trisurf(xs, ys, zs, linewidth=0.1, antialiased=True)
    plt.show()


def main():
    mesh_json = mload.load_mesh_json(Path("./circle_coarse_mesh.json"))
    mesh = mload.parse_json(mesh_json)
    q0 = initial_cond(mesh)
    sol = solve(10, mesh)
    plot(mesh, sol.y[-1])



if __name__ == "__main__":
    main()
