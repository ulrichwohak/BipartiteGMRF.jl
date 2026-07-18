# BipartiteGMRF.jl

`BipartiteGMRF.jl` fits Gaussian Markov random-field random-effects models on
bipartite graphs. The library API is table-in and typed-struct-out: callers
provide a `DataFrame`, choose a model type, solver, and weighting, and receive
a `GMRFResult`.

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

The library does not read or write Parquet, CSV, JSON, or `estimates.txt`
files. Application scripts can perform file I/O and pass a `DataFrame` to the
package.

## Quick Example

```julia
using BipartiteGMRF, DataFrames

df = DataFrame(
    firm_id = [1, 1, 2, 2],
    worker_id = [10, 11, 11, 12],
    y = [1.0, 0.8, 1.2, 0.7],
)

result = fit_mle(BipartiteNormalizedModel, df;
                 solver=ExactCholesky(), decompose=50, seed=42)

result.rho                    # local dependence parameter
coef(result)                  # [rho, sigma_a, sigma_z, sigma_epsilon]
loglikelihood(result)         # StatsAPI log-likelihood

fitted_vd = decompose(result; kind=:fitted, probes=50, seed=42)
block = cov_block(covariance(result; kind=:model); firms=[1], workers=[10])
```

See the [Examples](examples.md) page for the two-step
`suffstats` / `fit_mle` workflow and weighting options.
