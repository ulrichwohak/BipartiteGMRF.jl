# ═══════════════════════════════════════════════════════════════════════════
# Variance decomposition: unified API
# ═══════════════════════════════════════════════════════════════════════════

"""
    decompose(result::GMRFResult; kind=:model, probes=200, seed=42,
              target=result.stats.weighting.target, verbose=false)

Estimate the variance decomposition implied by a fitted bipartite GMRF.

Returns a `VarianceDecomposition` containing firm, worker, cross, residual,
and total variance components.

`kind` selects the decomposition type:
- `:model` — decomposition from the GMRF's precision structure alone
- `:fitted` — includes fitted effects (posterior mode + trace correction)
"""
function decompose(
    result::GMRFResult;
    kind::Symbol=:model,
    probes::Int=200,
    seed::Int=42,
    target::Symbol=result.stats.weighting.target,
    verbose::Bool=false,
)
    if kind == :model
        return _decompose_model(result; probes=probes, seed=seed, target=target, verbose=verbose)
    elseif kind == :fitted
        return _decompose_fitted(result; probes=probes, seed=seed, target=target, verbose=verbose)
    else
        throw(ArgumentError("decompose kind must be :model or :fitted; got $(kind)."))
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Covariance extraction: unified API
# ═══════════════════════════════════════════════════════════════════════════

"""
    covariance(result::GMRFResult; kind=:model, units=:original)

Factor the fitted precision matrix for covariance extraction.

`kind` selects the precision matrix:
- `:model` — the GMRF's precision Q
- `:fitted` — the data-augmented precision Q + λV'V

`units` may be `:original` or `:scaled`. Pass the returned
`CovarianceOperator` to `cov_block` to extract selected entity blocks.
"""
function covariance(
    result::GMRFResult;
    kind::Symbol=:model,
    units::Symbol=:original,
)
    if kind == :model
        return _covariance_model(result; units=units)
    elseif kind == :fitted
        return _covariance_fitted(result; units=units)
    else
        throw(ArgumentError("covariance kind must be :model or :fitted; got $(kind)."))
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# StatsAPI and accessors
# ═══════════════════════════════════════════════════════════════════════════

"""
    coef(result::GMRFResult)

Return fitted model coefficients as a named tuple.
"""
coef(result::GMRFResult) = (
    rho = result.rho,
    sigma_a = result.sigma_a,
    sigma_z = result.sigma_z,
    sigma_epsilon = result.sigma_epsilon,
    rho_eps = result.rho_eps,
)

"""
    loglikelihood(result::GMRFResult)

Return the fitted log-likelihood value.
"""
loglikelihood(result::GMRFResult) = -result.nll

"""
    nobs(result::GMRFResult)

Return the number of model observations used by the likelihood.
"""
nobs(result::GMRFResult) = result.stats.K

"""
    dof(result::GMRFResult)

Return the number of estimated parameters (ρ, σ_a, σ_z, σ_ε, and optionally ρ_ε).
"""
dof(result::GMRFResult) = result.rho_eps === nothing ? 4 : 5

"""
    aic(result::GMRFResult)

Akaike Information Criterion: -2logℓ + 2k.
"""
aic(result::GMRFResult) = -2.0 * loglikelihood(result) + 2.0 * dof(result)

"""
    bic(result::GMRFResult)

Bayesian Information Criterion: -2logℓ + k⋅log(n).
"""
bic(result::GMRFResult) = -2.0 * loglikelihood(result) + dof(result) * log(nobs(result))

isfitted(::GMRFResult) = true
islinear(::GMRFResult) = false

"""
    nll(result::GMRFResult)

Return the fitted negative log-likelihood objective value.
"""
nll(result::GMRFResult) = result.nll

"""
    converged(result::GMRFResult)

Return whether the optimizer reported convergence.
"""
converged(result::GMRFResult) = result.converged

function Base.show(io::IO, result::GMRFResult)
    print(io,
        "GMRFResult(rho=$(result.rho), sigma_a=$(result.sigma_a), ",
        "sigma_z=$(result.sigma_z), sigma_epsilon=$(result.sigma_epsilon), ",
        "nll=$(result.nll), converged=$(result.converged)",
    )
    result.model isa BipartiteVarianceStableModel &&
        print(io, ", rho_status=$(get(result.metadata, :rho_status, :unknown))")
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::GMRFResult)
    println(io, "GMRFResult")
    println(io, "  converged: ", result.converged)
    println(io, "  nll: ", result.nll)
    println(io, "  parameters:")
    println(io, "    rho: ", result.rho)
    if result.model isa BipartiteVarianceStableModel
        println(io, "    rho limit: ", rho_limit(result.model))
        println(io, "    rho utilization: ", get(result.metadata, :rho_utilization, NaN))
        println(io, "    rho status: ", get(result.metadata, :rho_status, :unknown))
    end
    println(io, "    sigma_a: ", result.sigma_a)
    println(io, "    sigma_z: ", result.sigma_z)
    println(io, "    sigma_epsilon: ", result.sigma_epsilon)
    result.rho_eps !== nothing && println(io, "    rho_eps: ", result.rho_eps)
    println(io, "  model:")
    println(io, "    model: ", nameof(typeof(result.model)))
    println(io, "    solver: ", nameof(typeof(result.solver)))
    println(io, "    observations: ", result.stats.K)
    println(io, "    person-year rows: ", result.stats.personyear_rows)
    println(io, "    firms: ", result.stats.N_firms)
    print(io, "    workers: ", result.stats.N_workers)
end
