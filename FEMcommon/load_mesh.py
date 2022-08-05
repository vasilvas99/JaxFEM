#!/usr/bin/python3


import jax
import jax.numpy as jnp
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Mesh:
    nodes: jnp.ndarray
    elements: jnp.ndarray
    elements_coords: jnp.ndarray
    dirichlet_nodes: jnp.ndarray


def load_mesh_json(file_path: Path):
    with open(file_path) as f:
        return json.load(f)


def parse_json(json_dict):
    nodes = jnp.array(json_dict["nodes"])
    elements = jnp.array(json_dict["elements"])
    dirichlet_nodes = jnp.array(json_dict["dirichletboundary"])

    # convert elements and dirichlet nodes to 0-based
    elements = jax.vmap(lambda element: jnp.array([coord - 1 for coord in element]))(
        elements
    )
    dirichlet_nodes = jax.vmap(lambda dnode: dnode - 1)(dirichlet_nodes)

    # pre-calculate elements coordinates
    # uses more memory but saves on trashing during assembly
    elements_coords = jax.vmap(
        lambda element: jnp.array([nodes[node_idx] for node_idx in element])
    )(elements)

    return Mesh(nodes, elements, elements_coords, dirichlet_nodes)


def main():
    mesh_dict = load_mesh_json(Path("./FEMcommon/test_mesh.json"))
    parsed_mesh = parse_json(mesh_dict)


if __name__ == "__main__":
    main()
