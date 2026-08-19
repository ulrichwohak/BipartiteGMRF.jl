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

A warm start on its own is not guaranteed to be faster, and the reasons are
worth knowing before reading too much into a timing.

### Size the simplex to match the start

Nelder-Mead does not begin at a point but at a simplex around it: vertex `j+1`
differs from the start in coordinate `j` only, and equals
`(1 + simplex_scale)·x_j + simplex_shift`. At the default `simplex_scale = 0.5`
that is a 50% relative perturbation — enormous on the unconstrained scale, where
`log σ = −0.9` reaches `−1.325`, i.e. `σ` down 35%. So a warm-started fit with
default settings still searches a wide region, and most of the benefit of `init`
is thrown away. Shrink the simplex to keep the search local:

```julia
# search ±5% around a good starting point
fit_mle(BipartiteVarianceStableModel, f, w, y;
        solver = ExactCholesky(simplex_scale = 0.05),
        init   = params(pilot))

# or a uniform absolute box, independent of coordinate magnitude
ExactCholesky(simplex_scale = 0.0, simplex_shift = 0.05)
```

`simplex_shift` is the absolute term, and it is what rescues coordinates near
zero: at exactly zero the relative term vanishes and only the shift remains. A
warm start produces exactly those coordinates — `log omega = 0`,
`atanh(eta) ≈ 0`, `log sigma ≈ 0` at `sigma ≈ 1`, `atanh(rho/rho_limit) ≈ 0` at
`rho ≈ 0` — so a simplex that is degenerate in one of them can satisfy the
convergence test having barely moved. Setting both knobs to zero is rejected for
that reason. `ExactCholesky` recovers through its L-BFGS polish; `HutchSLQ` has
no polish, so check that a warm-started `HutchSLQ` fit actually moved.

### The stopping tolerance also depends on the start

`g_reltol` is scaled by `max(1, |NLL(x₀)|)`, so a better starting point yields a
tighter absolute threshold. `ExactCholesky` floors it at `1e-3`, so it usually
does not move there; under `HutchSLQ` the scaling passes through and a better
start can cost extra iterations rather than fewer. (For a negative NLL the
direction reverses.) This is deliberate — pinning it would silently change when
existing fits stop.
