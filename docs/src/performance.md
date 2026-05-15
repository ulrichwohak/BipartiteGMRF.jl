# Performance

The package does not set global BLAS thread counts at load time. For many sparse
Cholesky and iterative-solver workloads, using one BLAS thread can improve
repeatability and avoid oversubscription:

```julia
using LinearAlgebra
BLAS.set_num_threads(1)
```

Make that choice in applications or scripts, not in package code.

For larger graphs, prefer `HutchSLQ()` and tune `logdet_probes`,
`lanczos_iters`, `cg_tol`, and `cg_maxiter` against the desired runtime and
stochastic tolerance.
