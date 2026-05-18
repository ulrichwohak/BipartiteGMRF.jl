# Priors

Priors describe the precision model for the latent firm and worker effects.
They are intentionally separate from solvers.

- `NormalizedPrior(; adjacency=:degree, prior_adjacency=:binary)`: default
  degree-normalized precision model.
- `UnnormalizedPrior(; prior_adjacency=:binary)`: paper-style `D - rho*A`
  precision model.
- `SpectralPrior(; prior_adjacency=:binary, seed=12345)`:
  spectral-normalized adjacency model. The seed controls the power-iteration
  initialization used for spectral normalization.
- `VarianceStablePrior()`: variance-stable precision model ported as a distinct
  prior.

## Supported Solver Matrix

`NormalizedPrior()` and `UnnormalizedPrior()` support `ExactCholesky()` and
`HutchSLQ()`. `SpectralPrior()` and `VarianceStablePrior()` currently support
`HutchSLQ()` only. Unsupported combinations throw `ArgumentError` before
fitting.
