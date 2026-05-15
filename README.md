# BipartiteGMRF

Julia tools for fitting bipartite-graph Gaussian Markov random-field random
effects models with prior and posterior variance decompositions and covariance
block extraction.

## Development

This repository now contains the reusable `BipartiteGMRF` package in `src/`.
The scripts under `src/estimate`, `src/create`, and `src/post_estimation` are
legacy project and reproducibility utilities; they are not part of the library
API and may depend on application-only packages such as `Parquet2`, `CSV`, or
`JSON`.

For local package work:

```julia
using Pkg
Pkg.activate(".")
Pkg.test()
```

The package API accepts `DataFrame` inputs. It deliberately does not read or
write Parquet, CSV, JSON, or `estimates.txt` files.

## Quick Start

```julia
using BipartiteGMRF, DataFrames

result = gmrf_mle(
    df;
    outcome=:y,
    firm_id=:firm_id,
    worker_id=:worker_id,
    prior=NormalizedPrior(),
    solver=ExactCholesky(),
    decompose=200,
    seed=42,
)

posterior = posterior_decomposition(result; probes=200, seed=42)
block = cov_block(prior_covariance(result); firms=[1, 2], workers=[10, 11])
```

## Priors And Solvers

The default prior is `NormalizedPrior()`, the degree-normalized bipartite GMRF.
`UnnormalizedPrior()` and `SpectralPrior()` are available for the corresponding
precision models. `VarianceStablePrior()` is exposed as a distinct prior model.

`ExactCholesky()` is deterministic and intended for moderate sparse problems.
`HutchSLQ()` uses PCG plus stochastic Lanczos quadrature for larger problems.
Unsupported combinations throw `ArgumentError` before fitting; for example,
`ExactCholesky()` currently does not support `SpectralPrior()` or
`VarianceStablePrior()`.

## Weighting

Observation weighting is configured with `Weighting`:

```julia
Weighting(observations=:raw)
Weighting(observations=:edge)
Weighting(observations=:effective, rho_eps=0.5)
Weighting(observations=:effective, rho_eps=:estimate)
```

Decomposition targets are `:estimation`, `:personyear`, and `:edge`.

## Covariance Blocks

Covariance extraction uses fitted results and preserved entity IDs:

```julia
prior_op = prior_covariance(result; units=:original)
post_op = posterior_covariance(result; units=:original)

firm_worker = cov_block(prior_op; row_firms=[1, 2], col_workers=[10, 11])
principal = cov_block(post_op; firms=[1, 2], workers=[10, 11])
```

`CovarianceBlock` stores the numeric matrix plus row/column entity metadata.
