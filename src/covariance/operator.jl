function _covariance_model(result::GMRFResult; units::Symbol=:original)
    units in (:original, :scaled) ||
        throw(ArgumentError("units must be :original or :scaled; got $(units)."))
    problem = result.problem
    sigma_a = result.sigma_a / problem.y_std
    sigma_z = result.sigma_z / problem.y_std
    Q = precision_matrix(problem, result.rho, sigma_a, sigma_z)
    return CovarianceOperator(:model, cholesky(Symmetric(Q)), result, units)
end

function _covariance_fitted(result::GMRFResult; units::Symbol=:original)
    units in (:original, :scaled) ||
        throw(ArgumentError("units must be :original or :scaled; got $(units)."))
    problem = result.problem
    sigma_a = result.sigma_a / problem.y_std
    sigma_z = result.sigma_z / problem.y_std
    sigma_e = result.sigma_epsilon / problem.y_std
    M = posterior_precision_matrix(problem, result.rho, sigma_a, sigma_z, sigma_e)
    return CovarianceOperator(:fitted, cholesky(Symmetric(M)), result, units)
end
