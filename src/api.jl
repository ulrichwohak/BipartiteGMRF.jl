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

coef(result::GMRFResult) = (
    rho = result.rho,
    sigma_a = result.sigma_a,
    sigma_z = result.sigma_z,
    sigma_epsilon = result.sigma_epsilon,
    rho_eps = result.rho_eps,
)

nll(result::GMRFResult) = result.nll
converged(result::GMRFResult) = result.converged

function Base.show(io::IO, result::GMRFResult)
    print(io,
        "GMRFResult(rho=$(result.rho), sigma_a=$(result.sigma_a), ",
        "sigma_z=$(result.sigma_z), sigma_epsilon=$(result.sigma_epsilon), ",
        "nll=$(result.nll), converged=$(result.converged))",
    )
end
