import jax
import jax.numpy as jnp
import numpy as onp
from dataclasses import dataclass
from typing import Callable
import scipy.integrate
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
    y = [problem.y0]

    #TODO: figure out how to compile loop body with jax
    for i in range(n-1):
        y.append(y[i] + step*problem.f(t[i], y[i]))

    return ODESolution(t, onp.array(y))

def scipy_ivp(problem: ODEProblem):
    sol = scipy.integrate.solve_ivp(problem.f, (problem.t0, problem.Tmax), problem.y0)
    return ODESolution(sol.t, sol.y)

def concentrate_mass_matrix(global_mass):
    return jnp.diag(global_mass.sum(1))

def main():
    test_mtx = jnp.array([[1,5,9],[3,8,1]])
    print(test_mtx)
    print(concentrate_mass_matrix(test_mtx))

if __name__ == "__main__":
    main()