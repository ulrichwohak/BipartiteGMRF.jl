# Models

Model types describe the precision structure of the latent firm and worker
effects. They are intentionally separate from solvers. All model types
subtype `AbstractBipartiteModel <: GaussianMarkovRandomFields.LatentModel`
and are constructed from a rectangular sparse adjacency matrix (firms ×
workers):

- `BipartiteNormalizedModel(A; rho_limit=0.99)`: default degree-normalized
  precision model.
- `BipartiteUnnormalizedModel(A; rho_limit=0.99)`: paper-style `D - rho*A`
  precision model.
- `BipartiteSpectralModel(A; seed=12345, rho_limit=0.99)`:
  spectral-normalized adjacency model. The seed controls the power-iteration
  initialization used for spectral normalization.
- `BipartiteVarianceStableModel(A; strict_forest=false, rho_limit=0.99)`:
  variance-stable precision model for forest-like graphs. Cyclic graphs warn
  by default; with `strict_forest=true` they throw `ArgumentError`. Pass
  `rho_limit=:auto` to opt into a non-backtracking feasibility limit resolved
  during `fit_mle`.

When fitting with `fit_mle(ModelType, df)` or `fit_mle(ModelType, ss)` the
model is built internally from the sufficient statistics' prior adjacency;
construct a model directly (and pass it to `fit_mle(model, ss)`) when you
need a custom adjacency or limit.

`rho_limit` sets the open interval `(-rho_limit, rho_limit)` used to transform
and validate the latent firm-worker dependence parameter. The default
`rho_limit=0.99` preserves the historical numerical optimization bound; it is
not a model-theoretic restriction.

For an automatic variance-stable limit, forests resolve to `0.99`; cyclic
graphs resolve to `min(0.99, 0.98 / lambda_nb)`. The fit always emits an
information message and records the spectrum, true ceiling, source, and active
limit in `result.stats.metadata`. Numeric limits remain explicit expert
overrides. Call `feasibility(model, stats)` to audit one; an unsafe limit
warns but is not changed. Every variance-stable fit reports its limit
utilization, and `rho_at_bound(result)` identifies a freely estimated `rho`
at 98% or more of the active limit.

## Direct GMRF Access

The model types implement the GaussianMarkovRandomFields.jl `LatentModel`
interface, so a parameterized model is one call away from a full GMRF:

```julia
gmrf = model(; ρ=0.5, σ_a=1.0, σ_z=0.8)
x = rand(gmrf)
v = GaussianMarkovRandomFields.var(gmrf)
```

## Supported Solver Matrix

`BipartiteNormalizedModel`, `BipartiteUnnormalizedModel`, and
`BipartiteVarianceStableModel` support `ExactCholesky()` and `HutchSLQ()`.
`BipartiteSpectralModel` supports `HutchSLQ()` only. On cyclic
variance-stable graphs, direct factorization can incur substantial fill-in;
choose the solver to match graph size and sparsity. Unsupported combinations
throw `ArgumentError` before fitting.
