# Non-Backtracking Walk Theory

The variance-stable graph kernel is

```math
B(\rho) = I - \rho A + \rho^2(D-I)
        = (1-\rho^2)I + \rho^2D - \rho A,
```

where `A` is the symmetric adjacency matrix of the bipartite prior graph and
`D` is its degree matrix. Let ``P_k`` count non-backtracking walks of length
``k`` between each pair of nodes. The initial matrices and recurrence are

```math
P_0=I,\qquad P_1=A,\qquad P_2=A^2-D,
```

```math
P_k=P_{k-1}A-P_{k-2}(D-I),\qquad k\geq 3.
```

Whenever the series converges, its generating function is

```math
\sum_{k=0}^{\infty}\rho^kP_k=(1-\rho^2)B(\rho)^{-1}.
```

This identity makes the interpretation of the variance-stable precision
direct: inverse-kernel entries aggregate all non-backtracking routes, weighted
by their lengths.

## Forest Identity

In a forest, a non-backtracking walk cannot revisit a node. There is exactly
one such walk from ``i`` to ``j`` when its length equals the graph distance
``d(i,j)``, and none at other lengths. Therefore

```math
(1-\rho^2)[B(\rho)^{-1}]_{ij}=\rho^{d(i,j)}.
```

In particular, ``(1-\rho^2)[B(\rho)^{-1}]_{ii}=1`` for every node, and the
implied correlation across every edge is exactly ``\rho``. Cycles add further
non-backtracking routes and are precisely what breaks this degree-independent
marginal-variance identity.

## Cycle Bands

On a simple cycle of length ``L``, walks from one node to another occupy length
bands congruent to the clockwise and counterclockwise distances modulo ``L``.
For adjacent nodes, summing those two geometric bands gives

```math
(1-\rho^2)[B(\rho)^{-1}]_{ii}
    = \frac{1+\rho^L}{1-\rho^L},
```

```math
\operatorname{Corr}(x_i,x_j)
    = \frac{\rho+\rho^{L-1}}{1+\rho^L}.
```

For `C4`, these reduce to variance inflation
``(1+\rho^4)/(1-\rho^4)`` and adjacent correlation
``(\rho+\rho^3)/(1+\rho^4)``. The cyclicity strata returned by
[`implied_correlations`](@ref) distinguish edges in the 2-core from edges
adjacent to it and edges in the remaining tree structure.

## Bethe-Hessian Connection

With the common Bethe-Hessian convention

```math
H(r)=(r^2-1)I-rA+D,
```

the variance-stable kernel satisfies ``B(\rho)=\rho^2H(1/\rho)``. The
Ihara-Bass determinant identity also gives

```math
\det(I-\rho\mathcal{B})
  =(1-\rho^2)^{m-n}\det B(\rho),
```

where ``\mathcal{B}`` is the directed-edge non-backtracking matrix, ``m`` is
the number of edges, and ``n`` is the number of nodes. Thus reciprocal
non-backtracking eigenvalues identify singularities of the kernel.

If ``\lambda_{NB}`` is the spectral radius of ``\mathcal{B}``, the guarded
feasible region is

```math
|\rho| < \min\left(1,\frac{1}{\lambda_{NB}}\right).
```

The inequality is strict. For forests, ``\lambda_{NB}=0`` and only
``|\rho|<1`` remains. [`VarianceStablePrior`](@ref) with `rho_limit=:auto`
stays inside this ceiling by using `0.98/lambda_NB` when a cyclic component
binds.
