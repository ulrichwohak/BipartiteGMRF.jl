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

## Inference

```@docs
stderror
vcov
confint
observed_information
with_standard_errors
```

`stderror`, `vcov`, and `confint` extend `StatsAPI`. Standard errors come from
the observed information (the numerical Hessian of the negative log-likelihood)
at the fitted parameters, with a delta-method transform to original outcome
units. For `HutchSLQ` the likelihood is stochastic, so these require an explicit
`compute_se=true` opt-in. Passing `compute_se=true` to [`gmrf_mle`](@ref) caches
the standard errors on the result so `show` displays them.

## Covariance

```@docs
CovarianceOperator
CovarianceBlock
prior_covariance
posterior_covariance
cov_block
```
