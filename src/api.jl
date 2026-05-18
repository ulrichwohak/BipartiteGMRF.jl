"""
    gmrf_mle(df; outcome=:y, firm_id=:firm_id, worker_id=:worker_id,
             prior=NormalizedPrior(), solver=ExactCholesky(),
             weighting=Weighting(), decompose=200, fix_rho=nothing,
             max_degree=nothing, standardize=true, on_missing=:drop,
             seed=42, verbose=false)

Construct and fit a bipartite-GMRF model from a `DataFrame`.

This is the high-level entry point: it prepares a `GMRFProblem`, fits it with
`solve`, and optionally computes a prior variance decomposition. The result is
a `GMRFResult` with parameters in original outcome units when
`standardize=true`.
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
    println(io, "GMRFResult")
    println(io, "  converged: ", result.converged)
    println(io, "  nll: ", result.nll)
    println(io, "  parameters:")
    println(io, "    rho: ", result.rho)
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
