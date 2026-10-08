# BipartiteGMRF.jl

`BipartiteGMRF.jl` fits Gaussian Markov random-field random-effects models on
bipartite graphs. The library API is arrays-in and typed-struct-out: callers
provide the graph as parallel firm-index / worker-index vectors together with
an outcome vector, choose a model type, solver, and weighting, and receive a
`GMRFResult`.

The core model is

```text
y_e = a_f(e) + z_w(e) + epsilon_e
```

with jointly Gaussian firm and worker effects. The package supports marginal
likelihood fitting, model and fitted variance decomposition, covariance block
extraction from fitted models, and simulation from the latent field.

Estimation follows the Distributions.jl `suffstats` / `fit_mle` convention —
the package extends those generic functions — and fitted results implement
`StatsAPI.StatisticalModel`.

## Package Boundary

The package is an estimator, not a data manipulator. It does not read or
write files, does not know about tables or column names, and does not drop
or repair observations. Application code maps entity identifiers to dense
1-based integer indices, filters unusable rows, and passes plain vectors.

## Quick Example

```julia
using BipartiteGMRF

f = [1, 1, 2, 2]        # firm index per observation
w = [1, 2, 2, 3]        # worker index per observation
y = [1.0, 0.8, 1.2, 0.7]

result = fit_mle(BipartiteNormalizedModel, f, w, y;
                 solver=ExactCholesky(), seed=42)

result.rho                    # local dependence parameter
coef(result)                  # [rho, sigma_a, sigma_z, sigma_epsilon]
loglikelihood(result)         # StatsAPI log-likelihood

fitted_vd = decompose(result; kind=:fitted, probes=50, seed=42)
block = cov_block(covariance(result; kind=:model); firms=[1], workers=[1])
```

See the [Examples](examples.md) page for the two-step
`suffstats` / `fit_mle` workflow and weighting options.
