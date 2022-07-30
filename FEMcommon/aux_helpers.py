import jax
import jax.numpy as jnp

def concentrate_mass_matrix(global_mass):
    return jnp.diag(global_mass.sum(1))


def main():
    test_mtx = jnp.array([[1,1,1],[2,2,2],[3,3,3]])
    print(test_mtx)
    print(concentrate_mass_matrix(test_mtx))

if __name__ == "__main__":
    main()