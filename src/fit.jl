# ═══════════════════════════════════════════════════════════════════════════
# fit_mle: standard MLE interface (Distributions.jl convention)
# ═══════════════════════════════════════════════════════════════════════════

"""
    fit_mle(::Type{M}, ss::BipartiteGMRFStats; solver=ExactCholesky(),
            rho_limit=0.99, decompose=nothing, fix_rho=nothing, seed=42,
            verbose=false, kwargs...)

Fit a bipartite GMRF model by maximum likelihood using precomputed sufficient statistics.

Returns a `GMRFResult`. This is the standard estimation entry point,
following the Distributions.jl `fit_mle(D, suffstats)` pattern.
"""
function fit_mle(
    ::Type{M},
    ss::BipartiteGMRFStats;
    solver::AbstractGMRFSolver=ExactCholesky(),
    rho_limit::Union{Float64,Symbol}=0.99,
    decompose::Union{Bool,Nothing,Int}=nothing,
    fix_rho::Union{Nothing,Float64}=nothing,
    seed::Int=42,
    strict_forest::Bool=false,
    verbose::Bool=false,
) where {M<:AbstractBipartiteModel}
    # Resolve VS auto rho_limit via NB spectrum
    vs_metadata = NamedTuple()
    resolved_rho_limit = rho_limit
    if M <: BipartiteVarianceStableModel && rho_limit isa Symbol
        rho_limit == :auto ||
            throw(ArgumentError("rho_limit symbol must be :auto; got $(rho_limit)."))
        spectrum = nb_spectrum(ss.A_prior)
        spectrum.converged || throw(ArgumentError(
            "Automatic VS feasibility could not be resolved because the non-backtracking eigensolver did not converge.",
        ))
        ceiling = nb_rho_ceiling(spectrum.lambda_nb)
        resolved_rho_limit = nb_recommended_limit(spectrum.lambda_nb)
        @info @sprintf(
            "VS feasibility resolved: lambda_NB=%.4f, rho_ceiling=%.4f, rho_limit=%.4f, source=auto",
            spectrum.lambda_nb,
            ceiling,
            resolved_rho_limit,
        )
        vs_metadata = (
            nb_spectrum=spectrum,
            rho_limit_source=:auto,
            rho_ceiling=ceiling,
            resolved_rho_limit=resolved_rho_limit,
        )
    elseif M <: BipartiteVarianceStableModel
        vs_metadata = (
            nb_spectrum=nothing,
            rho_limit_source=:explicit,
            rho_ceiling=nothing,
            resolved_rho_limit=Float64(rho_limit),
        )
    end

    # Construct the model directly from adjacency
    model = _build_model(M, ss.A_prior, Float64(resolved_rho_limit); seed=seed, strict_forest=strict_forest)

    # Construct GMRFProblem from stats + model
    problem = GMRFProblem(;
        y = ss.base_y,
        ydot = ss.ydot,
        projected_y = ss.projected_y,
        VtV = ss.VtV,
        A_obs = ss.A_obs,
        At_obs = ss.At_obs,
        cnt_f = ss.cnt_f,
        cnt_w = ss.cnt_w,
        firm_ids = ss.firm_ids,
        worker_ids = ss.worker_ids,
        firm_to_index = ss.firm_to_index,
        worker_to_index = ss.worker_to_index,
        base_f_rows = ss.base_f_rows,
        base_w_cols = ss.base_w_cols,
        base_y = ss.base_y,
        base_T = ss.base_T,
        decomp_f_rows = ss.decomp_f_rows,
        decomp_w_cols = ss.decomp_w_cols,
        decomp_y = ss.decomp_y,
        decomp_T = ss.decomp_T,
        N_firms = ss.N_firms,
        N_workers = ss.N_workers,
        K = ss.K,
        personyear_rows = ss.personyear_rows,
        y_mean = ss.y_mean,
        y_std = ss.y_std,
        standardize = ss.standardize,
        model = model,
        weighting = ss.weighting,
        rho_eps_likelihood = ss.rho_eps_likelihood,
        within_ss = ss.within_ss,
        within_df = ss.within_df,
        personyear_within_ss = ss.personyear_within_ss,
        log_weight_sum = ss.log_weight_sum,
        effective_weight_sum = ss.effective_weight_sum,
        effective_weight_over_T_sum = ss.effective_weight_over_T_sum,
        mean_effective_weight = ss.mean_effective_weight,
        max_effective_weight = ss.max_effective_weight,
        metadata = merge(ss.metadata, vs_metadata),
    )

    return solve(problem, solver; decompose=decompose, fix_rho=fix_rho, seed=seed, verbose=verbose)
end

"""
    fit_mle(::Type{M}, df::DataFrame; outcome=:y, firm_id=:firm_id, worker_id=:worker_id,
            solver=ExactCholesky(), rho_limit=0.99, weighting=Weighting(),
            model_adjacency=:binary, max_degree=nothing, standardize=true,
            on_missing=:drop, decompose=nothing, fix_rho=nothing, seed=42,
            verbose=false, kwargs...)

One-step convenience: compute sufficient statistics from a DataFrame and fit.
"""
function fit_mle(
    ::Type{M},
    df::DataFrame;
    outcome::Symbol=:y,
    firm_id::Symbol=:firm_id,
    worker_id::Symbol=:worker_id,
    solver::AbstractGMRFSolver=ExactCholesky(),
    rho_limit::Union{Float64,Symbol}=0.99,
    weighting::Weighting=Weighting(),
    model_adjacency::Symbol=:binary,
    max_degree::Union{Nothing,Int}=nothing,
    standardize::Bool=true,
    on_missing::Symbol=:drop,
    decompose::Union{Bool,Nothing,Int}=nothing,
    fix_rho::Union{Nothing,Float64}=nothing,
    seed::Int=42,
    strict_forest::Bool=false,
    verbose::Bool=false,
) where {M<:AbstractBipartiteModel}
    ss = suffstats(M, df;
        outcome=outcome,
        firm_id=firm_id,
        worker_id=worker_id,
        weighting=weighting,
        model_adjacency=model_adjacency,
        max_degree=max_degree,
        standardize=standardize,
        on_missing=on_missing,
    )
    return fit_mle(M, ss;
        solver=solver,
        rho_limit=rho_limit,
        decompose=decompose,
        fix_rho=fix_rho,
        seed=seed,
        strict_forest=strict_forest,
        verbose=verbose,
    )
end

# ─── Model construction dispatch ─────────────────────────────────────────

function _build_model(::Type{BipartiteNormalizedModel}, A::SparseMatrixCSC{Float64,Int}, rho_limit::Float64; kwargs...)
    return BipartiteNormalizedModel(A; rho_limit=rho_limit)
end

function _build_model(::Type{BipartiteUnnormalizedModel}, A::SparseMatrixCSC{Float64,Int}, rho_limit::Float64; kwargs...)
    return BipartiteUnnormalizedModel(A; rho_limit=rho_limit)
end

function _build_model(::Type{BipartiteSpectralModel}, A::SparseMatrixCSC{Float64,Int}, rho_limit::Float64; seed::Int=12345, kwargs...)
    return BipartiteSpectralModel(A; rho_limit=rho_limit, seed=seed)
end

function _build_model(::Type{BipartiteVarianceStableModel}, A::SparseMatrixCSC{Float64,Int}, rho_limit::Float64; strict_forest::Bool=false, kwargs...)
    return BipartiteVarianceStableModel(A; strict_forest=strict_forest, rho_limit=rho_limit)
end
