from pathlib import Path
from time import monotonic

import jax.numpy as jnp
import jax
from jax.config import config
config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import FEMcommon.aux_helpers as helpers
from FEMcommon.assemble_global import assemble_global_matrix, assemble_global_vector
from FEMcommon.local_matrices import local_stiffness, local_vector, local_mass
import FEMcommon.load_mesh as mload



def calculate_stiffness_matrix(mesh: mload.Mesh):
    gstiffness = assemble_global_matrix(mesh, local_stiffness)
    return jnp.array(gstiffness)


def calculate_mass_matrix(mesh: mload.Mesh):
    gmass = assemble_global_matrix(mesh, local_mass)
    gmass = helpers.concentrate_mass_matrix(gmass)
    return jnp.array(gmass)


def initial_cond(mesh: mload.Mesh, inital_temp, hot_radius):
    nodes = mesh.nodes
    return jnp.where(jnp.linalg.norm(nodes, ord=2, axis=1) <= hot_radius, inital_temp, 0)


def solve(initial_temp, hot_radius, Tmax, timestep, mesh: mload.Mesh):
    
    t0 = monotonic()
    q0 = initial_cond(mesh, initial_temp, hot_radius)
    mass = calculate_mass_matrix(mesh)
    stiffness = calculate_stiffness_matrix(mesh)
    system_mtx = - jnp.linalg.inv(mass) @ stiffness

    print(f"System matrix calculated in {monotonic()-t0}s. Starting ODE solution.")
    t0 = monotonic()
    def rhs(t, q):
        return system_mtx@q

    ode_p = helpers.ODEProblem(0, Tmax, rhs, q0)
    solution = helpers.implicit_euler(ode_p, timestep)
    print(f"ODE solution done in {monotonic()-t0}s.")
    return solution


def animate_plot(mesh, solution):
    # TODO: probably add animation controls
    print("Starting animation generation.")
    xs = mesh.nodes[:, 0]
    ys = mesh.nodes[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot first frame
    line = ax.plot_trisurf(xs, ys, solution.y[0], color='b')

    def data(i, line):
        # for every frame subsequent frame, clear the axes and re-plot
        zs = solution.y[i]
        ax.clear()
        ax.set_zlim(0, 100)
        line = ax.plot_trisurf(xs, ys, zs, linewidth=0.1, antialiased=True)
        return line

    ani = animation.FuncAnimation(
        fig, data, fargs=(line,),blit=False)
    print("Animation prepared. Rendering gif.")
    ani.save('./results/fick_law_animation.gif', writer='imagemagick', fps=5)
    print("Gif saved as: fick_law_animation.gif. Showing matplotlib interface.")
    plt.show()


def main():
    mesh_json = mload.load_mesh_json(Path("./test_meshes/circle_fine_mesh.json"))
    mesh = mload.parse_json(mesh_json)
    sol = solve(initial_temp=70, hot_radius=0.4, Tmax=10, timestep=0.1, mesh=mesh)
    animate_plot(mesh, sol)


if __name__ == "__main__":
    main()
