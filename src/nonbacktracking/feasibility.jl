function nb_rho_ceiling(lambda_nb::Float64)
    lambda_nb == 0.0 && return 1.0
    return min(1.0, 1.0 / lambda_nb)
end

function nb_recommended_limit(lambda_nb::Float64)
    lambda_nb == 0.0 && return 0.99
    return min(0.99, 0.98 / lambda_nb)
end


"""
    feasibility(problem; seed=12345, kwargs...)

Report variance-stable feasibility for a prepared problem. The result contains
the non-backtracking radius, the model-theoretic `rho_ceiling`, the active
numeric `rho_limit`, the recommended guarded limit, its source, and whether the
active limit is below the ceiling.

Automatic problems reuse their preparation-time spectrum. Explicit numeric
limits are audited without being changed; an unsafe explicit limit emits a
warning and remains active for estimation.
"""
function feasibility(problem::GMRFProblem; seed::Int=12345, kwargs...)
    problem.model isa BipartiteVarianceStableModel ||
        throw(ArgumentError("feasibility is defined only for BipartiteVarianceStableModel problems."))
    cached = get(problem.metadata, :nb_spectrum, nothing)
    spectrum = cached isa NBSpectrum ? cached : nb_spectrum(problem; seed=seed, kwargs...)
    source = get(problem.metadata, :rho_limit_source, :explicit)
    limit = rho_limit(problem.model)
    ceiling = spectrum.converged ? nb_rho_ceiling(spectrum.lambda_nb) : NaN
    recommended = spectrum.converged ? nb_recommended_limit(spectrum.lambda_nb) : NaN
    safe = spectrum.converged && limit < ceiling

    if !spectrum.converged
        @warn "VS feasibility could not be assessed because the non-backtracking eigensolver did not converge; the active limit is unchanged."
    elseif source == :explicit && !safe
        @warn @sprintf(
            "VS feasibility warning: explicit rho_limit=%.4f is not below rho_ceiling=%.4f (lambda_NB=%.4f); the limit is unchanged",
            limit,
            ceiling,
            spectrum.lambda_nb,
        )
    end
    return (
        spectrum=spectrum,
        lambda_nb=spectrum.lambda_nb,
        rho_ceiling=ceiling,
        rho_limit=limit,
        recommended_rho_limit=recommended,
        source=source,
        safe=safe,
    )
end

"""
    rho_at_bound(result)

Return `true` when a freely estimated variance-stable `rho` uses at least 98%
of its active optimization limit. Fixed-rho and non-variance-stable fits return
`false`.
"""
function rho_at_bound(result::GMRFResult)
    result.problem.model isa BipartiteVarianceStableModel || return false
    fixed = get(result.metadata, :fix_rho, nothing)
    fixed === nothing || return false
    return abs(result.rho) / rho_limit(result.problem.model) >= 0.98
end

function fit_result_metadata(
    problem::GMRFProblem,
    rho::Float64,
    fix_rho::Union{Nothing,Float64},
)
    base = (fix_rho=fix_rho,)
    problem.model isa BipartiteVarianceStableModel || return base

    limit = rho_limit(problem.model)
    utilization = abs(rho) / limit
    at_bound = fix_rho === nothing && utilization >= 0.98
    status = fix_rho !== nothing ? :fixed : at_bound ? :bound_censored : :interior
    message = @sprintf(
        "VS fit status: rho=%.4f, rho_limit=%.4f, utilization=%.4f, status=%s",
        rho,
        limit,
        utilization,
        String(status),
    )
    at_bound ? (@warn message) : (@info message)
    return merge(base, (
        rho_limit=limit,
        rho_utilization=utilization,
        rho_status=status,
        rho_at_bound=at_bound,
    ))
end
