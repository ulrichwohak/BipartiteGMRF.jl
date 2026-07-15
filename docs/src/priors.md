# Priors

Priors describe the precision model for the latent firm and worker effects.
They are intentionally separate from solvers.

- `NormalizedPrior(; adjacency=:degree, prior_adjacency=:binary,
  rho_limit=0.99)`: default degree-normalized precision model.
- `UnnormalizedPrior(; prior_adjacency=:binary, rho_limit=0.99)`:
  paper-style `D - rho*A` precision model.
- `SpectralPrior(; prior_adjacency=:binary, seed=12345, rho_limit=0.99)`:
  spectral-normalized adjacency model. The seed controls the power-iteration
  initialization used for spectral normalization.
- `VarianceStablePrior(; strict_forest=false, rho_limit=0.99)`:
  variance-stable precision model ported as a distinct prior. Cyclic graphs
  warn by default; with `strict_forest=true` they throw `ArgumentError`.

`rho_limit` sets the open interval `(-rho_limit, rho_limit)` used to transform
and validate the latent firm-worker dependence parameter. The default
`rho_limit=0.99` preserves the historical numerical optimization bound; it is
not a model-theoretic restriction.

## Supported Solver Matrix

`NormalizedPrior()`, `UnnormalizedPrior()`, and `VarianceStablePrior()` support
`ExactCholesky()` and `HutchSLQ()`. `SpectralPrior()` supports `HutchSLQ()`
only. On cyclic variance-stable graphs, direct factorization can incur
substantial fill-in; choose the solver to match graph size and sparsity.
Unsupported combinations throw `ArgumentError` before fitting.
