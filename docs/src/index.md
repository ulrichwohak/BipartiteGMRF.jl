# BipartiteGMRF.jl

`BipartiteGMRF.jl` fits Gaussian Markov random-field random-effects models on
bipartite graphs. The library API is table-in and typed-struct-out: callers
provide a `DataFrame`, configure priors, solvers, and weighting, and receive a
`GMRFResult`.

The core model is

```text
y_e = a_f(e) + z_w(e) + epsilon_e
```

with jointly Gaussian firm and worker effects. The package supports marginal
likelihood fitting, prior and posterior variance decomposition, and covariance
block extraction from fitted models.

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

result = gmrf_mle(df; solver=ExactCholesky(), decompose=50, seed=42)
post = posterior_decomposition(result; probes=50, seed=42)
block = cov_block(prior_covariance(result); firms=[1], workers=[10])
```
