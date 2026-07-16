"""
    gmrf_mle(df; outcome=:y, firm_id=:firm_id, worker_id=:worker_id,
             prior=NormalizedPrior(), solver=ExactCholesky(),
             weighting=Weighting(), decompose=200, fix_rho=nothing,
             max_degree=nothing, standardize=true, on_missing=:drop,
             seed=42, verbose=false)

Construct and fit a bipartite-GMRF model from a `DataFrame`.

This is the high-level entry point: it prepares a `GMRFProblem`, fits it with
`solve`, and optionally computes a model variance decomposition.
"""
function gmrf_mle(
    df::DataFrame;
    outcome::Symbol=:y,
    firm_id::Symbol=:firm_id,
    worker_id::Symbol=:worker_id,
    prior::AbstractGMRFPrior=NormalizedPrior(),
    solver::AbstractGMRFSolver=ExactCholesky(),
    weighting::Weighting=Weighting(),
    decompose::Union{Bool,Nothing,Int}=200,
    fix_rho::Union{Nothing,Float64}=nothing,
    max_degree::Union{Nothing,Int}=nothing,
    standardize::Bool=true,
    on_missing::Symbol=:drop,
    seed::Int=42,
    verbose::Bool=false,
)
    problem = GMRFProblem(
        df;
        outcome=outcome,
        firm_id=firm_id,
        worker_id=worker_id,
        prior=prior,
        weighting=weighting,
        max_degree=max_degree,
        standardize=standardize,
        on_missing=on_missing,
        verbose=verbose,
    )
    return solve(problem, solver; decompose=decompose, fix_rho=fix_rho, seed=seed, verbose=verbose)
end

# ═══════════════════════════════════════════════════════════════════════════
# Variance decomposition: unified API
# ═══════════════════════════════════════════════════════════════════════════

"""
    decompose(result::GMRFResult; kind=:model, probes=200, seed=42,
              target=result.problem.weighting.target, verbose=false)

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
    target::Symbol=result.problem.weighting.target,
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
nobs(result::GMRFResult) = result.problem.K

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
    result.prior isa VarianceStablePrior &&
        print(io, ", rho_status=$(get(result.metadata, :rho_status, :unknown))")
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::GMRFResult)
    println(io, "GMRFResult")
    println(io, "  converged: ", result.converged)
    println(io, "  nll: ", result.nll)
    println(io, "  parameters:")
    println(io, "    rho: ", result.rho)
    if result.prior isa VarianceStablePrior
        println(io, "    rho limit: ", rho_limit(result.prior))
        println(io, "    rho utilization: ", get(result.metadata, :rho_utilization, NaN))
        println(io, "    rho status: ", get(result.metadata, :rho_status, :unknown))
    end
    println(io, "    sigma_a: ", result.sigma_a)
    println(io, "    sigma_z: ", result.sigma_z)
    println(io, "    sigma_epsilon: ", result.sigma_epsilon)
    result.rho_eps !== nothing && println(io, "    rho_eps: ", result.rho_eps)
    println(io, "  model:")
    println(io, "    prior: ", nameof(typeof(result.prior)))
    println(io, "    solver: ", nameof(typeof(result.solver)))
    println(io, "    observations: ", result.problem.K)
    println(io, "    person-year rows: ", result.problem.personyear_rows)
    println(io, "    firms: ", result.problem.N_firms)
    print(io, "    workers: ", result.problem.N_workers)
end
