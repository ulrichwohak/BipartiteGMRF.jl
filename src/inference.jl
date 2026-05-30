# Observed-information standard errors and Wald inference for a fitted GMRF.
#
# The optimizer works on an unconstrained free-parameter vector (rho via
# limit*tanh, the sigmas via exp, rho_eps via a scaled logistic). Standard
# errors come from the numerical Hessian of the negative log-likelihood at the
# fitted point (the observed information), with a diagonal delta-method
# transform back to the natural, original-unit parameters.

# rho_eps is estimated only under effective weighting with `rho_eps=:estimate`.
function _estimate_rho_eps(problem::GMRFProblem)
    return problem.weighting.observations == :effective &&
           problem.weighting.rho_eps == :estimate
end

# Names of the estimated (free) parameters, in observed-information / vcov order.
function _free_names(result::GMRFResult)
    fix_rho = get(result.metadata, :fix_rho, nothing)
    names = Symbol[]
    fix_rho === nothing && push!(names, :rho)
    push!(names, :sigma_a, :sigma_z, :sigma_epsilon)
    _estimate_rho_eps(result.problem) && push!(names, :rho_eps)
    return names
end

# The fitted free unconstrained parameter vector. `theta_unconstrained` is the
# full vector (rho slot always present); drop it when rho was fixed.
function _free_point(result::GMRFResult)
    pfull = result.theta_unconstrained
    fix_rho = get(result.metadata, :fix_rho, nothing)
    return fix_rho === nothing ? collect(pfull) : pfull[2:end]
end

# Rebuild the exact free-parameter NLL objective used during optimization so the
# observed information can be evaluated at (and around) the fitted point.
function _free_objective(result::GMRFResult; seed::Int=42)
    problem = result.problem
    solver = result.solver
    fix_rho = get(result.metadata, :fix_rho, nothing)
    limit = rho_limit(problem.prior)
    estimate_rho_eps = _estimate_rho_eps(problem)
    cache = solver isa HutchSLQ ? make_hutch_cache(problem, solver) : nothing
    return function (pfree)
        pfull = full_params(Vector{Float64}(pfree), fix_rho, estimate_rho_eps; rho_limit=limit)
        stats = objective_stats(problem, pfull)
        if solver isa ExactCholesky
            return nll_exact_value(problem, pfull, stats)
        else
            return nll_hutch_value(problem, solver, pfull, stats, cache; seed=seed)
        end
    end
end

# Diagonal delta-method Jacobian factor d(natural, original units)/d(free
# unconstrained), aligned with `_free_names`. The transforms are elementwise, so
# the Jacobian is diagonal and each factor depends only on its own parameter.
function _free_jacobian(result::GMRFResult)
    names = _free_names(result)
    limit = rho_limit(result.prior)
    jac = map(names) do nm
        if nm === :rho
            limit * (1.0 - (result.rho / limit)^2)
        elseif nm === :rho_eps
            r = result.rho_eps::Float64
            r * (1.0 - r / 0.999)
        else  # sigma_a, sigma_z, sigma_epsilon: d(exp(theta)*y_std)/dtheta = sigma
            getfield(result, nm)::Float64
        end
    end
    return names, jac
end

# Build a full coef-shaped NamedTuple, filling non-estimated parameters with
# `nothing` (fixed rho, or rho_eps when not estimated).
function _named_over_params(names::Vector{Symbol}, vals::AbstractVector)
    d = Dict{Symbol,Any}(zip(names, vals))
    return (
        rho = get(d, :rho, nothing),
        sigma_a = d[:sigma_a],
        sigma_z = d[:sigma_z],
        sigma_epsilon = d[:sigma_epsilon],
        rho_eps = get(d, :rho_eps, nothing),
    )
end

# Numerical Hessian of the NLL with respect to the estimated free parameters,
# with the HutchSLQ stochastic-noise opt-in gate. Returns the symmetrized
# Hessian and the fitted free-parameter vector.
function _free_hessian(result::GMRFResult; compute_se::Bool, seed::Int)
    solver = result.solver
    if solver isa HutchSLQ
        compute_se || throw(ArgumentError(
            "Standard errors for HutchSLQ derive from a numerical Hessian of a " *
            "stochastic (SLQ) likelihood and inherit Monte-Carlo noise. Pass " *
            "`compute_se=true` to opt in (consider averaging across seeds), or " *
            "refit with ExactCholesky for deterministic standard errors."))
        @warn "HutchSLQ standard errors use a numerical Hessian of a stochastic " *
              "likelihood; values are noisy and depend on `seed`." seed
    end
    obj = _free_objective(result; seed=seed)
    pfree = _free_point(result)
    H = finite_difference_hessian(obj, pfree)
    return 0.5 .* (H .+ transpose(H)), pfree
end

# Estimator covariance on the free (unconstrained) scale: the inverse observed
# information, defined only when that information is positive definite. A non-PD
# information signals a non-converged or unidentified fit (e.g. a parameter at a
# boundary); warn and return NaNs rather than silently reporting zero or garbage
# standard errors.
function _free_cov(H::AbstractMatrix; warn::Bool=true)
    n = size(H, 1)
    V = try
        inv(cholesky(Symmetric(H)))
    catch
        nothing
    end
    if V === nothing || !all(isfinite, V)
        warn && @warn "Observed information is not positive definite; the fit may " *
                      "be non-converged or have unidentified parameters (check " *
                      "`converged`). Affected standard errors are returned as NaN."
        return fill(NaN, n, n)
    end
    return Matrix(V)
end

# Standard errors (original units) as a coef-shaped NamedTuple, from a free-scale
# covariance and the diagonal delta-method Jacobian.
function _stderror_from(result::GMRFResult, Vfree::AbstractMatrix)
    names, jac = _free_jacobian(result)
    se_free = [d < 0 ? NaN : sqrt(d) for d in diag(Vfree)]
    return _named_over_params(names, abs.(jac) .* se_free)
end

"""
    observed_information(result::GMRFResult; compute_se=false, seed=42)

Observed Fisher information at the fitted parameters: the numerical Hessian of
the negative log-likelihood with respect to the estimated *unconstrained*
parameters (the ones actually optimized). Rows/columns follow the order rho (if
estimated), `sigma_a`, `sigma_z`, `sigma_epsilon`, `rho_eps` (if estimated).

For `HutchSLQ` the likelihood is stochastic, so the Hessian is noisy; pass
`compute_se=true` to opt in. `seed` controls the SLQ probes for `HutchSLQ` and
is ignored for `ExactCholesky`.
"""
function observed_information(result::GMRFResult; compute_se::Bool=false, seed::Int=42)
    H, _ = _free_hessian(result; compute_se=compute_se, seed=seed)
    return H
end

"""
    vcov(result::GMRFResult; compute_se=false, seed=42)

Covariance matrix of the estimator for the natural parameters in original
outcome units, obtained by a delta-method transform of the inverse observed
information. Rows/columns follow the order rho (if estimated), `sigma_a`,
`sigma_z`, `sigma_epsilon`, `rho_eps` (if estimated). Extends `StatsAPI.vcov`.

See [`observed_information`](@ref) for the `compute_se`/`seed` semantics.
"""
function vcov(result::GMRFResult; compute_se::Bool=false, seed::Int=42)
    H, _ = _free_hessian(result; compute_se=compute_se, seed=seed)
    Vfree = _free_cov(H)
    _, jac = _free_jacobian(result)
    Vnat = similar(Vfree)
    @inbounds for j in axes(Vfree, 2), i in axes(Vfree, 1)
        Vnat[i, j] = jac[i] * Vfree[i, j] * jac[j]
    end
    return Vnat
end

"""
    stderror(result::GMRFResult; compute_se=false, seed=42)

Standard errors for the fitted parameters in original outcome units, returned as
a named tuple mirroring [`coef`](@ref): `rho`, `sigma_a`, `sigma_z`,
`sigma_epsilon`, `rho_eps`. Parameters that were not estimated (a fixed rho, or
`rho_eps` when not estimated) are `nothing`. Extends `StatsAPI.stderror`.

See [`observed_information`](@ref) for the `compute_se`/`seed` semantics.
"""
function stderror(result::GMRFResult; compute_se::Bool=false, seed::Int=42)
    H, _ = _free_hessian(result; compute_se=compute_se, seed=seed)
    return _stderror_from(result, _free_cov(H))
end

"""
    confint(result::GMRFResult; level=0.95, compute_se=false, seed=42)

Wald confidence intervals for the fitted parameters, returned as a named tuple
mirroring [`coef`](@ref) where each estimated entry is an `(lower, upper)`
tuple. Intervals are constructed on the unconstrained (tanh/log) scale and
back-transformed, so they always respect the natural parameter constraints
(`|rho| < rho_limit`, positive sigmas, `0 <= rho_eps < 1`). Non-estimated
parameters are `nothing`. Extends `StatsAPI.confint`.

See [`observed_information`](@ref) for the `compute_se`/`seed` semantics.
"""
function confint(result::GMRFResult; level::Real=0.95, compute_se::Bool=false, seed::Int=42)
    0.0 < level < 1.0 || throw(ArgumentError("level must lie in (0, 1); got $(level)."))
    H, pfree = _free_hessian(result; compute_se=compute_se, seed=seed)
    Vfree = _free_cov(H)
    names = _free_names(result)
    se_free = [d < 0 ? NaN : sqrt(d) for d in diag(Vfree)]
    z = norm_quantile(0.5 + level / 2.0)
    limit = rho_limit(result.prior)
    y_std = result.problem.y_std
    intervals = Vector{Tuple{Float64,Float64}}(undef, length(names))
    for (k, nm) in enumerate(names)
        lo = pfree[k] - z * se_free[k]
        hi = pfree[k] + z * se_free[k]
        intervals[k] = if nm === :rho
            (limit * tanh(lo), limit * tanh(hi))
        elseif nm === :rho_eps
            (rhoeps_from_unconstrained(lo), rhoeps_from_unconstrained(hi))
        else
            (exp(lo) * y_std, exp(hi) * y_std)
        end
    end
    return _named_over_params(names, intervals)
end

# Best-effort standard errors for display; never throws and skips the stochastic
# HutchSLQ path so that `show` stays cheap and side-effect free.
function _display_stderror(result::GMRFResult)
    result.solver isa ExactCholesky || return nothing
    return try
        H, _ = _free_hessian(result; compute_se=false, seed=42)
        _stderror_from(result, _free_cov(H; warn=false))
    catch
        nothing
    end
end

_format_estimate(value, se) = se === nothing ? string(value) : string(value, " ± ", se)
