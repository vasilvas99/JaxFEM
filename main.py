import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

import matplotlib.pyplot as plt

import FEMcommon.load_mesh as mload
from FEMcommon.local_matrices import local_stiffness, local_vector
from FEMcommon.assemble_global import assemble_global_matrix, assemble_global_vector
from pathlib import Path



def calculate_stiffness_matrix(mesh: mload.Mesh):
    gstiffness = assemble_global_matrix(mesh, local_stiffness)
    n = gstiffness.shape[0]
    # apply dirichlet b.c.s
    for i in range(n):
        for boundary_node in mesh.dirichlet_nodes:
            gstiffness[i, boundary_node] = 0
            gstiffness[boundary_node, i] = 0
    return jnp.array(gstiffness)

def calculate_load_vector(mesh: mload.Mesh):
    gvect = assemble_global_vector(mesh, local_vector)
    # apply dirichlet b.c.s
    for boundary_node in mesh.dirichlet_nodes:
        gvect[boundary_node] = 0  
    return jnp.array(gvect)

def plot(mesh, values):
    xs = mesh.nodes[:,0]
    ys = mesh.nodes[:, 1]
    zs = values
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(xs, ys, zs, linewidth=0.1, antialiased=True)
    plt.savefig("plot.png", dpi=1200)
    plt.show()

def main():
    mesh_json = mload.load_mesh_json(Path("./test_meshes/circle_fine_mesh.json"))
    mesh = mload.parse_json(mesh_json)
    gstiffness = calculate_stiffness_matrix(mesh)
    gvect = calculate_load_vector(mesh)
    q = jnp.linalg.solve(gstiffness, gvect)
    q = jnp.nan_to_num(q, 0)
    plot(mesh, q)


if __name__ == "__main__":
    main()
