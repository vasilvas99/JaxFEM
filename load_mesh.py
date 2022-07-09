import jax
import jax.numpy as jnp
import json
from pathlib import Path

from yaml import parse

def load_mesh_json(file_path: Path):
    with open(file_path) as f:
        return json.load(f)
    
def parse_json(json_dict):
    nodes = jnp.array(json_dict["nodes"])    
    elements = jnp.array(json_dict["elements"])
    dirichlet_nodes = jnp.array(json_dict["dirichletboundary"])

    #TODO: convert elements and dirichlet nodes to 0-based

    elements_coords = jax.vmap(lambda element: jnp.array([nodes[node_idx] for node_idx in element]))(elements)
    
    
if __name__ == "__main__":
    mesh_dict = load_mesh_json(Path("./test_mesh.json"))
    parse_json(mesh_dict)