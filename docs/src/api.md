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

```@docs
suffstats(::Type{<:AbstractBipartiteModel}, ::DataFrame)
fit_mle(::Type{<:AbstractBipartiteModel}, ::BipartiteGMRFStats)
fit_mle(::Type{<:AbstractBipartiteModel}, ::DataFrame)
fit_mle(::AbstractBipartiteModel, ::BipartiteGMRFStats)
solve(::AbstractBipartiteModel, ::BipartiteGMRFStats, ::ExactCholesky)
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
