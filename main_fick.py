from pathlib import Path

import jax.numpy as jnp
import jax
from jax.config import config
config.update("jax_enable_x64", True)

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
    system_mtx = onp.array(system_mtx)

    # System mtx is assembled correctly. The ode solver should be the problem then ...
    def rhs(t, q):
        # print(f'{q=}')
        # print(f'{q.shape}')
        # print(f'{system_mtx.shape}')
        # exit(-1)
        return system_mtx@q

    ode_p = helpers.ODEProblem(0, Tmax, rhs, q0)
    solution = helpers.explicit_euler(ode_p, 0.0001)
    print("ODE solution done")
    return solution

def animate_plot(mesh, solution):
    #TODO: probably add animation controls
    xs = mesh.nodes[:, 0]
    ys = mesh.nodes[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot first frame
    line = ax.plot_trisurf(xs, ys, solution.y[0],color= 'b')
    def data(i, line):
        #for every frame subsequent frame, clear the axes and re-plot
        zs = solution.y[i]
        ax.clear()
        ax.set_zlim(0, 100)
        line = ax.plot_trisurf(xs, ys, zs, linewidth=0.1, antialiased=True)
        return line

    ani = animation.FuncAnimation(fig, data, fargs=(line,), interval=120, blit=False)
    ani.save('fick_law_animation.gif',writer='imagemagick',fps=5)
    plt.show()

        

def main():
    mesh_json = mload.load_mesh_json(Path("./test_meshes/regular_mesh.json"))
    mesh = mload.parse_json(mesh_json)
    q0 = initial_cond(mesh)
    sol = solve(10, mesh)
    print(f'Final system state: {sol.y[-1]}')
    animate_plot(mesh, sol)


if __name__ == "__main__":
    main()
