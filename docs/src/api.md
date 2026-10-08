# API

```@meta
CurrentModule = BipartiteGMRF
```

## Estimation

The estimation entry points extend the
[Distributions.jl](https://juliastats.org/Distributions.jl/stable/fit/)
`suffstats` / `fit_mle` generic functions, so they compose with
`using Distributions` without name clashes. `BipartiteGMRFStats` subtypes
`Distributions.SufficientStats`.

Data enters as three parallel vectors — firm index, worker index, outcome —
with 1-based dense integer indices on each side. The package is an estimator,
not a data manipulator: mapping entity identifiers to indices and filtering
unusable rows happen before the data reaches it.

```@docs
suffstats(::Type{<:AbstractBipartiteModel}, ::AbstractVector{<:Integer}, ::AbstractVector{<:Integer}, ::AbstractVector{<:Real})
fit_mle(::Type{<:AbstractBipartiteModel}, ::BipartiteGMRFStats)
fit_mle(::Type{<:AbstractBipartiteModel}, ::AbstractVector{<:Integer}, ::AbstractVector{<:Integer}, ::AbstractVector{<:Real})
fit_mle(::AbstractBipartiteModel, ::BipartiteGMRFStats)
solve(::AbstractBipartiteModel, ::BipartiteGMRFStats, ::AbstractGMRFSolver)
BipartiteGMRFStats
GMRFResult
```

## Models

```@docs
AbstractBipartiteModel
BipartiteNormalizedModel
BipartiteUnnormalizedModel
BipartiteSpectralModel
BipartiteVarianceStableModel
BipartiteGraph
Weighting
```

## Solvers

```@docs
AbstractGMRFSolver
ExactCholesky
HutchSLQ
```

## StatsAPI Interface

`GMRFResult` implements `StatsAPI.StatisticalModel`. `coef`, `coefnames`,
`loglikelihood`, `nobs`, `dof`, `aic`, `bic`, `isfitted`, and `islinear`
extend StatsAPI; `params` extends the Distributions.jl generic; `converged`
extends `Optim.converged`.

```@docs
coef(::GMRFResult)
coefnames(::GMRFResult)
params(::GMRFResult)
loglikelihood(::GMRFResult)
nobs(::GMRFResult)
dof(::GMRFResult)
aic(::GMRFResult)
bic(::GMRFResult)
isfitted(::GMRFResult)
islinear(::GMRFResult)
nll
converged(::GMRFResult)
```

## Decompositions

```@docs
decompose
VarianceDecomposition
```

## Covariance

```@docs
covariance
cov_block
CovarianceOperator
CovarianceBlock
```

## Simulation

```@docs
simulate
```

## Non-Backtracking Diagnostics

```@docs
NBSpectrum
nb_spectrum
feasibility
rho_at_bound
```
