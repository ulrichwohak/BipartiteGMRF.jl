function _covariance_model(result::GMRFResult; units::Symbol=:original)
    units in (:original, :scaled) ||
        throw(ArgumentError("units must be :original or :scaled; got $(units)."))
    model = result.model
    stats = result.stats
    sigma_a = result.sigma_a / stats.y_std
    sigma_z = result.sigma_z / stats.y_std
    Q = model_precision(model, result.rho, sigma_a, sigma_z)
    return CovarianceOperator(:model, cholesky(Symmetric(Q)), result, units)
end

function _covariance_fitted(result::GMRFResult; units::Symbol=:original)
    units in (:original, :scaled) ||
        throw(ArgumentError("units must be :original or :scaled; got $(units)."))
    model = result.model
    stats = result.stats
    sigma_a = result.sigma_a / stats.y_std
    sigma_z = result.sigma_z / stats.y_std
    sigma_e = result.sigma_epsilon / stats.y_std
    M = fitted_precision(model, stats.VtV, result.rho, sigma_a, sigma_z, sigma_e)
    return CovarianceOperator(:fitted, cholesky(Symmetric(M)), result, units)
end
