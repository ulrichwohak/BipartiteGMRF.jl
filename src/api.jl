"""
    gmrf_mle(df; outcome=:y, firm_id=:firm_id, worker_id=:worker_id,
             prior=NormalizedPrior(), solver=ExactCholesky(),
             weighting=Weighting(), decompose=200, fix_rho=nothing,
             max_degree=nothing, standardize=true, on_missing=:drop,
             seed=42, compute_se=false, verbose=false)

Construct and fit a bipartite-GMRF model from a `DataFrame`.

This is the high-level entry point: it prepares a `GMRFProblem`, fits it with
`solve`, and optionally computes a prior variance decomposition. The result is
a `GMRFResult` with parameters in original outcome units when
`standardize=true`.

Pass `compute_se=true` to also compute and cache observed-information standard
errors, which makes `show` print them in a regression-table style; standard
errors are otherwise available on demand via [`stderror`](@ref).

See also [`coef`](@ref), [`stderror`](@ref), [`confint`](@ref),
[`loglikelihood`](@ref), [`nobs`](@ref), [`converged`](@ref).
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
    compute_se::Bool=false,
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
    return solve(problem, solver;
        decompose=decompose, fix_rho=fix_rho, seed=seed, compute_se=compute_se, verbose=verbose)
end

"""
    coef(result::GMRFResult)

Return fitted model coefficients as a named tuple.

The tuple contains `rho`, `sigma_a`, `sigma_z`, `sigma_epsilon`, and `rho_eps`.
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
        "nll=$(result.nll), converged=$(result.converged))",
    )
end

function Base.show(io::IO, ::MIME"text/plain", result::GMRFResult)
    se = _display_stderror(result)
    se_of(name) = se === nothing ? nothing : getfield(se, name)
    rho_fixed = get(result.metadata, :fix_rho, nothing) !== nothing
    println(io, "GMRFResult")
    println(io, "  converged: ", result.converged)
    println(io, "  nll: ", result.nll)
    println(io, "  parameters:")
    println(io, "    rho: ", rho_fixed ? string(result.rho, " (fixed)") :
                             _format_estimate(result.rho, se_of(:rho)))
    println(io, "    sigma_a: ", _format_estimate(result.sigma_a, se_of(:sigma_a)))
    println(io, "    sigma_z: ", _format_estimate(result.sigma_z, se_of(:sigma_z)))
    println(io, "    sigma_epsilon: ", _format_estimate(result.sigma_epsilon, se_of(:sigma_epsilon)))
    result.rho_eps !== nothing &&
        println(io, "    rho_eps: ", _format_estimate(result.rho_eps, se_of(:rho_eps)))
    println(io, "  model:")
    println(io, "    prior: ", nameof(typeof(result.prior)))
    println(io, "    solver: ", nameof(typeof(result.solver)))
    println(io, "    observations: ", result.problem.K)
    println(io, "    person-year rows: ", result.problem.personyear_rows)
    println(io, "    firms: ", result.problem.N_firms)
    print(io, "    workers: ", result.problem.N_workers)
end
