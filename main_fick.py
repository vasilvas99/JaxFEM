from pathlib import Path

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as onp


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

    print("Matrix calculations done, moving onto solving ODE")

    def rhs(t, q):
        return system_mtx@q

    ode_p = helpers.ODEProblem(0, Tmax, rhs, q0)
    solution = helpers.explicit_euler(ode_p, 0.005)
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


def animate_plot(mesh, solution):
    xs = mesh.nodes[:, 0]
    ys = mesh.nodes[:, 1]
    
    ax = plt.axes(projection='3d')
    ax.set_zlim(0,100)
    ax.plot_trisurf(xs, ys, zs, linewidth=0.1, antialiased=True)

    def update_plot(frame_number, zarray, plot):
        plot[0].remove()
        plot[0] = ax.plot_trisurf(xs, ys, zarray[:,:,frame_number], linewidth=0.1, antialiased=True)

    ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(zarray, plot), interval=1000/fps)

        

def main():
    mesh_json = mload.load_mesh_json(Path("./circle_coarse_mesh.json"))
    mesh = mload.parse_json(mesh_json)
    q0 = initial_cond(mesh)
    sol = solve(1, mesh)
    plot(mesh, sol.y[-1])



if __name__ == "__main__":
    main()
