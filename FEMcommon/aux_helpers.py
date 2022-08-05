import jax
import jax.numpy as jnp
import numpy as onp
from dataclasses import dataclass
from typing import Callable
from scipy.optimize import fsolve, root


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

    # TODO: figure out how to compile loop body with jax
    for i in range(n - 1):
        y.append(y[i] + step * problem.f(t[i], y[i]))

    return ODESolution(t, onp.array(y))


def RK4(problem: ODEProblem, step):
    t = onp.arange(problem.t0, problem.Tmax, step)
    n = t.shape[0]
    y = [problem.y0] * n
    f = problem.f
    for i in range(n - 1):
        k1 = step * (f(t[i], y[i]))
        k2 = step * (f((t[i] + step / 2), (y[i] + k1 / 2)))
        k3 = step * (f((t[i] + step / 2), (y[i] + k2 / 2)))
        k4 = step * (f((t[i] + step), (y[i] + k3)))
        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y[i + 1] = y[i] + k
    return ODESolution(t, onp.array(y))


def implicit_euler(problem: ODEProblem, step):
    t = onp.arange(problem.t0, problem.Tmax, step)
    n = t.shape[0]
    y = [problem.y0] * n
    f = problem.f

    for i in range(n - 1):

        def roots_of(yc):
            return yc - (y[i] + step * f(t[i + 1], yc))

        y[i + 1] = root(roots_of, y[i], method="krylov").x

    return ODESolution(t, onp.array(y))


def concentrate_mass_matrix(global_mass):
    return jnp.diag(global_mass.sum(1))


def main():
    test_mtx = jnp.array([[1, 5, 9], [3, 8, 1]])
    print(test_mtx)
    print(concentrate_mass_matrix(test_mtx))


if __name__ == "__main__":
    main()
