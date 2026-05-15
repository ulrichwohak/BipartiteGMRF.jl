# Covariance

Covariance extraction factors the fitted prior or posterior precision matrix
and extracts requested entity blocks by batched solves.

- `prior_covariance(result; units=:original)`: factor fitted prior precision.
- `posterior_covariance(result; units=:original)`: factor fitted posterior
  precision.
- `cov_block(op; ...)`: extract principal or rectangular blocks by firm and
  worker IDs.
- `CovarianceOperator`: cached factorization and metadata for extraction.

Examples:

```julia
prior_op = prior_covariance(result; units=:original)
post_op = posterior_covariance(result; units=:scaled)

principal = cov_block(prior_op; firms=[1, 2], workers=[10, 11])
rect = cov_block(post_op; row_firms=[1], col_workers=[10, 11])
```
