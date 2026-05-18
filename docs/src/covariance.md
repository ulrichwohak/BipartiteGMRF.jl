# Covariance

Covariance extraction factors the fitted prior or posterior precision matrix
and extracts requested entity blocks by batched solves.

- `prior_covariance(result; units=:original)`: factor fitted prior precision.
- `posterior_covariance(result; units=:original)`: factor fitted posterior
  precision.
- `cov_block(op; ..., batch_size=16)`: extract principal or rectangular blocks
  by firm and worker IDs.
- `CovarianceOperator`: cached factorization and metadata for extraction.

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
prior_op = prior_covariance(result; units=:original)
post_op = posterior_covariance(result; units=:scaled)

principal = cov_block(prior_op; firms=[1, 2], workers=[10, 11])
rect = cov_block(post_op; row_firms=[1], col_workers=[10, 11])
```
