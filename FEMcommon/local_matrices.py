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


shapes = [s1, s2, s3]
grad_shapes = [jax.grad(shape) for shape in shapes]


def isoparametric_coord(verts, p, i):
    f_eval = jnp.array(list(map(lambda f: f(p), shapes)))
    return verts[:, i].dot(f_eval)


def real_coords(verts, p):
    real_coords = [isoparametric_coord(verts, p, i) for i in range(len(p))]
    return jnp.array(real_coords)


real_coords_jac = jax.jacfwd(real_coords, argnums=1)


def triangle_quadrature(verts, integrand):
    quadrature_weights = jnp.array([1/6, 1/6, 1/6])
    quadrature_nodes = jnp.array([[1/2, 0], [1/2, 1/2], [0, 1/2]])

    vals_at_nodes = jax.vmap(
        lambda node: integrand(verts, node))(quadrature_nodes)
    return quadrature_weights*vals_at_nodes.sum(0)


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

def right_vector_field_2D(point):
    x, y  = point
    return 20*jnp.array([[1, -y]]) # row vector

def local_advection_integrand(verts, p):
    f_eval = jnp.array([list(map(lambda f: f(p), shapes))]).T
    jac = real_coords_jac(verts, p).T
    jac_inv = jnp.linalg.inv(jac)
    b = right_vector_field_2D(real_coords(verts, p))
    grad_shape_eval = jnp.array(list(map(lambda f: f(p), grad_shapes))).T 
    return (f_eval@b@jac_inv@grad_shape_eval)*jnp.linalg.det(jac)

def local_advection(verts):
    lai = local_advection_integrand
    return triangle_quadrature(verts, lai)

local_mass = jax.jit(local_mass)
local_stiffness = jax.jit(local_stiffness)
local_advection = jax.jit(local_advection)
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


def test_advection():
    v = jnp.array([[0,0],[1,0],[0,1]])
    p = jnp.array([0.1,0.0])
    print(right_vector_field_2D(real_coords(v, p)))
    print(right_vector_field_2D(p))
    print(f'{local_advection(v)=}')


def main():
    n = 5_000_000
    delta = test_vmap(n)
    print(f'Time taken to run {n} samples {delta}')
    print(f'Single run time {delta/n}')


if __name__ == "__main__":
    test_advection()
