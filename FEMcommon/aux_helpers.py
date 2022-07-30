import jax
import jax.numpy as jnp
import numpy as onp
from dataclasses import dataclass
from typing import Callable

@dataclass
class ODESolution:
    t: onp.ndarray
    y: onp.ndarray

@dataclass
class ODEProblem:
    t0: float
    Tmax: float
    f: Callable
    y0: onp.ndarray

def explicit_euler(problem: ODEProblem, step):
    t = onp.arange(problem.t0, problem.Tmax, step)
    n = t.shape[0]
    y = onp.array(onp.repeat(problem.y0[None,:], n, axis=0))
    #TODO: figure out how to compile loop body with jax
    for i in range(1, n-1):
        y[i+1] = y[i] + step*problem.f(t, y[i])

    return ODESolution(t, y)

def concentrate_mass_matrix(global_mass):
    return jnp.diag(global_mass.sum(1))

def main():
    test_mtx = jnp.array([[1,5,9],[3,8,1]])
    print(test_mtx)
    print(concentrate_mass_matrix(test_mtx))

if __name__ == "__main__":
    main()