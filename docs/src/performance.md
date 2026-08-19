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

## Warm starts

The default starting point is a fixed heuristic — `sigma_a = 0.7`,
`sigma_z = 0.04`, `sigma_epsilon = 0.4` on a unit-variance outcome — and is not
adapted to the data. `init` replaces it, field by field, in the outcome's own
units:

```julia
r  = fit_mle(BipartiteVarianceStableModel, f, w, y)
r2 = fit_mle(BipartiteVarianceStableModel, f, w, y;
             error_groups = firm, init = params(r))
```

Two uses. The first is cost: several error models fitted on the same graph can
share one cheap pilot estimate instead of each climbing from the same
dataset-agnostic default. The second is diagnostic: when a fit lands on a
boundary — `rho` pinned at `rho_limit`, a variance collapsed to zero — refitting
from a different, informed starting point is what separates a genuine feature of
the likelihood from an artifact of where the optimizer began.

A warm start is not guaranteed to be faster, and the reasons are worth knowing
before reading too much into a timing:

- Nelder-Mead's initial simplex perturbs each coordinate by 50%, so it still
  spans a wide region around `init`.
- Coordinates whose *unconstrained* value is near zero — `log omega = 0`,
  `atanh(eta) ≈ 0`, `log sigma ≈ 0` at `sigma ≈ 1`, all of which a warm start
  tends to produce — get an absolute step of 0.025 instead. A simplex that is
  near-degenerate in one coordinate can satisfy the convergence test almost
  immediately. `ExactCholesky` recovers through its L-BFGS polish; `HutchSLQ`
  has no polish, so check that a warm-started `HutchSLQ` fit actually moved.
- The convergence threshold is scaled by the objective *at the starting point*,
  so a better start also buys a tighter tolerance. Under `ExactCholesky` a floor
  masks this; under `HutchSLQ` it can cost extra iterations.

Exposing the simplex scale is tracked as issue #115.
