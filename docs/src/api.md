# API

The high-level entry point is `gmrf_mle`. For repeated solver runs on the same
data, build a `GMRFProblem` once and call `solve`.

## Core Calls

- `gmrf_mle(df; ...)`: construct a `GMRFProblem`, fit it, and optionally compute
  the prior variance decomposition.
- `GMRFProblem(df; ...)`: prepare data once with configurable column names,
  priors, weighting, missing handling, standardization, and max-degree
  filtering.
- `solve(problem, solver; ...)`: fit a prepared problem with `ExactCholesky()` or
  `HutchSLQ()`.
- `coef(result)`, `nll(result)`, and `converged(result)`: lightweight result
  accessors.

## Result Types

- `GMRFResult`: fitted parameters, diagnostics, decompositions, and the prepared
  problem.
- `VarianceDecomposition`: prior or posterior variance components.
- `CovarianceBlock`: extracted covariance matrix plus row/column metadata.
