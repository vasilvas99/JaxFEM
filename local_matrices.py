#!/usr/bin/python3
from random import random
from time import monotonic
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


def x_y(verts, p):
    return jnp.array([isoparametric_coord(verts, p, 0), isoparametric_coord(verts, p, 1)])


xy_jitted = jax.jit(x_y)
xy_jac = jax.jacfwd(xy_jitted, argnums=1)


def triangle_quadrature(verts, integrand):
    weights = jnp.array([1/6, 1/6, 1/6])
    nodes = jnp.array([[1/2, 0], [1/2, 1/2], [0, 1/2]])
    integral = weights[0]*integrand(verts, nodes[0])
    for idx, node in enumerate(nodes[1:]):
        integral += weights[idx]*integrand(verts, node)
    return integral


def local_mass_integrand(verts, p):
    f_eval = jnp.array(list(map(lambda f: f(p), shapes)))
    return jnp.outer(f_eval.T, f_eval) * jnp.linalg.det(xy_jac(verts, p))


local_mass_integrand_jitted = jax.jit(local_mass_integrand)


def local_mass(verts):
    lmi = local_mass_integrand_jitted
    return triangle_quadrature(verts, lmi)


def local_stiffness_integrand(verts, p):
    grad_shape_eval_T = jnp.array(list(map(lambda f: f(p), grad_shapes)))
    jac_T = xy_jac(verts, p)
    jac_inv_T = jnp.linalg.inv(jac_T)
    return (grad_shape_eval_T@jac_inv_T@jac_inv_T.T@grad_shape_eval_T.T)*jnp.linalg.det(jac_T)


local_stiffness_integrand_jitted = jax.jit(local_stiffness_integrand)


def local_stiffness(verts):
    lsi = local_stiffness_integrand_jitted
    return triangle_quadrature(verts, lsi)


local_mass = jax.jit(local_mass)
local_stiffness = jax.jit(local_stiffness)


def test(n):
    data = jnp.array([[[random(), random()],[random(), random()],[random(), random()]] for x in range(n)])
    print("Starting execution")
    t = monotonic()
    for d in data:
        local_mass(d)
        local_stiffness(d)
    return monotonic() - t

def test_vmap(n):
    data = jnp.array([[[random(), random()],[random(), random()],[random(), random()]] for x in range(n)])
    print("Starting execution")
    t = monotonic()
    def execute(d):
        local_mass(d)
        local_stiffness(d)    
    jax.vmap(execute)(data)
    return monotonic() - t

if __name__ == "__main__":
    n = 5_000_000
    delta = test_vmap(n)
    print(f'Time taken to run {n} samples {delta}')
    print(f'Single run time {delta/n}')
