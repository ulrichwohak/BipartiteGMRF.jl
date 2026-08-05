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

Return the fitted parameters as a vector `[rho, sigma_a, sigma_z, sigma_epsilon]`,
with `rho_eps` appended when it was part of the likelihood. Labels are available
from [`coefnames`](@ref); a named view from [`params`](@ref).
"""
function coef(result::GMRFResult)
    c = [result.rho, result.sigma_a, result.sigma_z, result.sigma_epsilon]
    result.rho_eps === nothing || push!(c, result.rho_eps)
    return c
end

"""
    coefnames(result::GMRFResult)

Return the names of the entries of [`coef`](@ref).
"""
function coefnames(result::GMRFResult)
    names = ["rho", "sigma_a", "sigma_z", "sigma_epsilon"]
    result.rho_eps === nothing || push!(names, "rho_eps")
    return names
end

"""
    params(result::GMRFResult)

Return the fitted parameters as a named tuple
`(rho=..., sigma_a=..., sigma_z=..., sigma_epsilon=..., rho_eps=...)`,
where `rho_eps` is `nothing` unless effective weighting was used.
"""
params(result::GMRFResult) = (
    rho = result.rho,
    sigma_a = result.sigma_a,
    sigma_z = result.sigma_z,
    sigma_epsilon = result.sigma_epsilon,
    rho_eps = result.rho_eps,
)

# Gaussian dimensions entering the fitted likelihood: person-year rows for
# :raw and :effective (full-data likelihoods), collapsed edges for :edge
# (edge-mean likelihood).
_loglikelihood_dims(stats::BipartiteGMRFStats) =
    stats.weighting.observations == :edge ? stats.K : stats.personyear_rows

"""
    loglikelihood(result::GMRFResult)

Return the maximized log-likelihood of the (mean-centered) data in original
outcome units, including the Gaussian normalizing constant and the Jacobian of
the internal standardization. Unlike [`nll`](@ref) — the raw optimizer
objective — this value is invariant to the `standardize` option and comparable
across fits of the same data.
"""
function loglikelihood(result::GMRFResult)
    n = _loglikelihood_dims(result.stats)
    return -(result.nll + n * log(result.stats.y_std) + 0.5 * n * log(2.0 * pi))
end

"""
    nobs(result::GMRFResult)

Return the number of model observations used by the likelihood.
"""
nobs(result::GMRFResult) = result.stats.K

"""
    dof(result::GMRFResult)

Return the number of *estimated* parameters. The baseline (ρ, σ_a, σ_z, σ_ε)
counts four; a `fix_rho` fit counts one fewer, and a jointly estimated
`rho_eps=:estimate` counts one more. Parameters held fixed do not count.
"""
function dof(result::GMRFResult)
    k = 4
    get(result.metadata, :fix_rho, nothing) === nothing || (k -= 1)
    w = result.stats.weighting
    w.observations == :effective && w.estimate_rho_eps && (k += 1)
    return k
end

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

"""
    isfitted(result::GMRFResult)

Return `true`; a `GMRFResult` always represents a completed fit.
"""
isfitted(::GMRFResult) = true

"""
    islinear(result::GMRFResult)

Return `false`; the bipartite GMRF likelihood is not linear in its parameters.
"""
islinear(::GMRFResult) = false

"""
    nll(result::GMRFResult)

Return the fitted value of the internal optimizer objective: the negative
log-likelihood of the standardized data with additive constants dropped.
Use [`loglikelihood`](@ref) for the constant-complete, original-units value.
"""
nll(result::GMRFResult) = result.nll

"""
    converged(result::GMRFResult)

Return whether the optimizer reported convergence. Extends `Optim.converged`.
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
