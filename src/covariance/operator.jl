"""
    prior_covariance(result::GMRFResult; units=:original)

Factor the fitted prior precision matrix for covariance extraction.

`units` may be `:original` or `:scaled`. Pass the returned
`CovarianceOperator` to `cov_block` to extract selected entity blocks.
"""
function prior_covariance(result::GMRFResult; units::Symbol=:original)
    units in (:original, :scaled) ||
        throw(ArgumentError("units must be :original or :scaled; got $(units)."))
    problem = result.problem
    sigma_a = result.sigma_a / problem.y_std
    sigma_z = result.sigma_z / problem.y_std
    Q = precision_matrix(problem, result.rho, sigma_a, sigma_z)
    return CovarianceOperator(:prior, cholesky(Symmetric(Q)), result, units)
end

"""
    posterior_covariance(result::GMRFResult; units=:original)

Factor the fitted posterior precision matrix for covariance extraction.

`units` may be `:original` or `:scaled`. The returned `CovarianceOperator`
caches the factorization used by `cov_block`.
"""
function posterior_covariance(result::GMRFResult; units::Symbol=:original)
    units in (:original, :scaled) ||
        throw(ArgumentError("units must be :original or :scaled; got $(units)."))
    problem = result.problem
    sigma_a = result.sigma_a / problem.y_std
    sigma_z = result.sigma_z / problem.y_std
    sigma_e = result.sigma_epsilon / problem.y_std
    M = posterior_precision_matrix(problem, result.rho, sigma_a, sigma_z, sigma_e)
    return CovarianceOperator(:posterior, cholesky(Symmetric(M)), result, units)
end
