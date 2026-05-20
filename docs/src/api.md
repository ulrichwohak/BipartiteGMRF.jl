# API

## Core Calls

```@docs
gmrf_mle
GMRFProblem
solve
coef
loglikelihood
nobs
nll
converged
```

`coef`, `loglikelihood`, and `nobs` extend `StatsAPI`.

## Priors And Weighting

```@docs
AbstractGMRFPrior
NormalizedPrior
UnnormalizedPrior
SpectralPrior
VarianceStablePrior
Weighting
```

## Solvers

```@docs
AbstractGMRFSolver
ExactCholesky
HutchSLQ
```

## Results And Decompositions

```@docs
GMRFResult
VarianceDecomposition
prior_decomposition
posterior_decomposition
```

## Covariance

```@docs
CovarianceOperator
CovarianceBlock
prior_covariance
posterior_covariance
cov_block
```
