# Implementation plans

## Advection matrix:

Let $ b = (b_1, b_2, b_3 ...)$ be a row-vector  (a vector valued function) represeting a fluid's velocity field.
Then, the local advection matrix can be written as (in index form):

$$
C_{i, j}^\kappa = (\bar{b} \cdot \nabla \phi_j, \phi_i)_\kappa
$$

For the global basis. 

Equivalently:

$$
C_{i, j}^\kappa = \iint_\kappa \bar{b} \cdot \nabla \phi_j \phi_i d\kappa
$$

In matrix form:

$$
C^\kappa = \iint_\kappa \Phi^T (\bar{b}\nabla \Phi)  d\kappa
$$

Where $\Phi = (\phi_1, \phi_2, ...) $ - row vector with the non-zero over $\kappa$ basis functions (from the global basis).

Changing to the standart element $E$ we obtain:

$$
C^\kappa = \iint_E \Psi^T (\bar{b} J ^{-1} \nabla \Psi) |J| dE
$$

Finally, for the local advection matrix we obtain:

$$
C^\kappa = |J| \iint_E \Psi^T b J ^{-1} \nabla \Psi dE
$$

The integral is then approximated with a quadrature formula and the global matrix is assembled as usual.

### Sanity check:

Let the problem be $n$-dimensional.

Let the number of shape functions be $m$.

$dim (\Psi^T) = (m, 1)$ - column vector

$dim(b) = (1,n)$ - row vector

$dim(J^{-1}) = (n,n)$ - matrix

$dim(\nabla \Psi) = (n , m)$ - matrix

Therefore:

$(m, 1) \times (1,n) \times (n,n) \times (n , m) \rightarrow (m, m)$
