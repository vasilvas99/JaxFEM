#!/usr/bin/python3

import jax
import load_mesh as mload
import numpy as np
from local_matrices import local_mass, local_vector


def calculate_local_matrices(mesh: mload.Mesh, local_matrix):
    return jax.vmap(lambda element: local_matrix(element))(mesh.elements_coords)


def assemble_global_matrix(mesh: mload.Mesh, local_matrix):
    n = mesh.nodes.shape[0]
    global_mtx = np.zeros((n, n))
    local_matrices = calculate_local_matrices(mesh, local_matrix)

    for idx, lm in enumerate(local_matrices):
        m = lm.shape[0]
        elements = mesh.elements
        for i in range(m):
            for j in range(m):
                global_mtx[elements[idx, i], elements[idx, j]] += lm[i, j]
    return global_mtx


def assemble_global_vector(mesh: mload.Mesh, local_vector):
    n = mesh.nodes.shape[0]
    global_vector = np.zeros(n)
    local_vectors = calculate_local_matrices(mesh, local_vector)
    for idx, lv in enumerate(local_vectors):
        m = lv.shape[0]
        elements = mesh.elements
        for i in range(m):
            global_vector[elements[idx, i]] += lv[i]
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
