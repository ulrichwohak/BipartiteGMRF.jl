# Covariance

Covariance extraction factors the fitted model or data-augmented precision
matrix and extracts requested entity blocks by batched solves.

- `covariance(result; kind=:model, units=:original)`: factor the fitted
  model precision `Q`.
- `covariance(result; kind=:fitted, units=:original)`: factor the
  data-augmented precision `Q + λV'V`.
- `cov_block(op; ..., batch_size=16)`: extract principal or rectangular blocks
  by firm and worker IDs.
- `CovarianceOperator`: cached factorization and metadata for extraction.

`units=:original` reports covariances in original outcome units;
`units=:scaled` reports them in internal standardized units.

## Batch Size And Memory

`cov_block` extracts selected columns of the inverse precision matrix by solving
against batches of unit vectors. `batch_size` controls how many requested
columns are solved at once. Larger batches can reduce sparse factor solve
overhead, but allocate a dense temporary right-hand-side workspace with
`n_latents * batch_size` entries, in addition to the returned covariance block.
Lower `batch_size` when extracting from large fitted graphs under tight memory
limits.

Examples:

```julia
model_op = covariance(result; kind=:model, units=:original)
fitted_op = covariance(result; kind=:fitted, units=:scaled)

principal = cov_block(model_op; firms=[1, 2], workers=[10, 11])
rect = cov_block(fitted_op; row_firms=[1], col_workers=[10, 11])
```
