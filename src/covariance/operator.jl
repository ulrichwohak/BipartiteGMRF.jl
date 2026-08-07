function _covariance_model(result::GMRFResult; units::Symbol=:original)
    units in (:original, :scaled) ||
        throw(ArgumentError("units must be :original or :scaled; got $(units)."))
    p = scaled_params(result)
    Q = model_precision(result.model, p.rho, p.sigma_a, p.sigma_z)
    return CovarianceOperator(:model, cholesky(Symmetric(Q)), result, units)
end

function _covariance_fitted(result::GMRFResult; units::Symbol=:original)
    units in (:original, :scaled) ||
        throw(ArgumentError("units must be :original or :scaled; got $(units)."))
    p = scaled_params(result)
    M = fitted_precision(
        result.model, result.stats.design.VtV,
        p.rho, p.sigma_a, p.sigma_z, p.sigma_epsilon,
    )
    return CovarianceOperator(:fitted, cholesky(Symmetric(M)), result, units)
end
