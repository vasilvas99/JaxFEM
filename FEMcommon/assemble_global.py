#!/usr/bin/python3

import jax
import FEMcommon.load_mesh as mload
import numpy as onp
from FEMcommon.local_matrices import local_mass, local_vector
import FEMcommon.fem_toolkit as rust_toolkit

def calculate_local_matrices(mesh: mload.Mesh, local_matrix):
    return jax.vmap(lambda element: local_matrix(element))(mesh.elements_coords)


def assemble_global_matrix(mesh: mload.Mesh, local_matrix):
    n = mesh.nodes.shape[0]
    global_mtx = onp.zeros((n, n))
    local_matrices = calculate_local_matrices(mesh, local_matrix)
    global_mtx = rust_toolkit.assemble_global_matrix(onp.array(global_mtx, dtype=onp.double),
                                                     onp.array(local_matrices, dtype=onp.double),
                                                     onp.array(mesh.elements, dtype=onp.int64))
    return global_mtx

def assemble_global_vector(mesh: mload.Mesh, local_vector):
    #TODO: figure out how to compile loop body with jax

    n = mesh.nodes.shape[0]
    global_vector = onp.zeros(n)
    local_vectors = calculate_local_matrices(mesh, local_vector)
    global_vector = rust_toolkit.assemble_global_vector(onp.array(global_vector, dtype=onp.double), 
                                                        onp.array(local_vectors, dtype=onp.double), 
                                                        onp.array(mesh.elements, dtype=onp.int64))
    return global_vector



def main():
    from pathlib import Path
    mesh_dict = mload.load_mesh_json(Path("./FEMcommon/test_mesh.json"))
    parsed_mesh = mload.parse_json(mesh_dict)
    gmass = assemble_global_matrix(parsed_mesh, local_mass)
    gvec = assemble_global_vector(parsed_mesh, local_vector)
    print(gmass)
    print(gvec)


if __name__ == "__main__":
    main()
