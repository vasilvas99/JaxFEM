#!/usr/bin/python3

from random import random
import jax
import jax.numpy as jnp


def s1(p):
    return 1 - p[0] - p[1]


def s2(p):
    return p[0]


def s3(p):
    return p[1]


shapes = [jax.jit(s1), jax.jit(s2), jax.jit(s3)]
grad_shapes = [jax.grad(shape) for shape in shapes]


def isoparametric_coord(verts, p, i):
    f_eval = jnp.array(list(map(lambda f: f(p), shapes)))
    return verts[:, i].dot(f_eval)


def real_coords(verts, p):
    real_coords = [isoparametric_coord(verts, p, i) for i in range(len(p))]
    return jnp.array(real_coords)


real_coords_jac = jax.jacfwd(real_coords, argnums=1)


def triangle_quadrature(verts, integrand):
    weights = jnp.array([1/6, 1/6, 1/6])
    nodes = jnp.array([[1/2, 0], [1/2, 1/2], [0, 1/2]])
    integral = weights[0]*integrand(verts, nodes[0])
    for idx, node in enumerate(nodes[1:]):
        integral += weights[idx]*integrand(verts, node)
    return integral


def local_mass_integrand(verts, p):
    f_eval = jnp.array(list(map(lambda f: f(p), shapes)))
    return jnp.outer(f_eval.T, f_eval) * jnp.linalg.det(real_coords_jac(verts, p))


def local_mass(verts):
    lmi = local_mass_integrand
    return triangle_quadrature(verts, lmi)


def local_stiffness_integrand(verts, p):
    grad_shape_eval = jnp.array(list(map(lambda f: f(p), grad_shapes)))
    jac = real_coords_jac(verts, p)
    jac_inv = jnp.linalg.inv(jac)
    return (grad_shape_eval@jac_inv@jac_inv.T@grad_shape_eval.T)*jnp.linalg.det(jac)


def local_stiffness(verts):
    lsi = local_stiffness_integrand
    return triangle_quadrature(verts, lsi)


def local_vector_integrand(verts, p):
    f_eval = jnp.array(list(map(lambda f: f(p), shapes)))
    jac = real_coords_jac(verts, p)
    return f_eval*jnp.linalg.det(jac)


def local_vector(verts):
    lvi = local_vector_integrand
    return triangle_quadrature(verts, lvi)


local_mass = jax.jit(local_mass)
local_stiffness = jax.jit(local_stiffness)
local_vector = jax.jit(local_vector)


def test(n):
    from time import monotonic
    data = jnp.array([[[random(), random()], [random(), random()], [
                     random(), random()]] for x in range(n)])
    print("Starting execution")
    t = monotonic()
    for d in data:
        local_mass(d)
        local_stiffness(d)
    return monotonic() - t


def test_vmap(n):
    from time import monotonic
    data = jnp.array([[[random(), random()], [random(), random()], [
                     random(), random()]] for x in range(n)])
    print("Starting execution")
    t = monotonic()

    def execute(d):
        local_mass(d)
        local_stiffness(d)
    jax.vmap(execute)(data)
    return monotonic() - t


def main():
    n = 5_000_000
    delta = test_vmap(n)
    print(f'Time taken to run {n} samples {delta}')
    print(f'Single run time {delta/n}')


if __name__ == "__main__":
    main()
