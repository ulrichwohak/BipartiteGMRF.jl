# ═══════════════════════════════════════════════════════════════════════════
# fit_mle: standard MLE interface (Distributions.jl convention)
# ═══════════════════════════════════════════════════════════════════════════

"""
    fit_mle(::Type{M}, ss::BipartiteGMRFStats; solver=ExactCholesky(),
            rho_limit=0.99, decompose=nothing, fix_rho=nothing, seed=42,
            strict_forest=false, verbose=false)

Fit a bipartite GMRF model by maximum likelihood using precomputed sufficient statistics.

Returns a [`GMRFResult`](@ref). This extends `Distributions.fit_mle` and follows its
`fit_mle(D, suffstats)` pattern.

`decompose` controls the optional model variance decomposition attached to the
result: `nothing` (default) skips it, a positive integer runs it with that many
probes. `rho_limit=:auto` (variance-stable model only) resolves the limit from
the non-backtracking spectrum.
"""
function fit_mle(
    ::Type{M},
    ss::BipartiteGMRFStats;
    solver::AbstractGMRFSolver=ExactCholesky(),
    rho_limit::Union{Float64,Symbol}=0.99,
    decompose::Union{Nothing,Int}=nothing,
    fix_rho::Union{Nothing,Float64}=nothing,
    seed::Int=42,
    strict_forest::Bool=false,
    verbose::Bool=false,
) where {M<:AbstractBipartiteModel}
    if rho_limit isa Symbol && !(M <: BipartiteVarianceStableModel)
        throw(ArgumentError(
            "rho_limit=:$(rho_limit) is only supported for BipartiteVarianceStableModel; pass a numeric rho_limit for $(nameof(M)).",
        ))
    end
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

    # Merge VS metadata into stats metadata
    merged_stats = isempty(vs_metadata) ? ss : _with_merged_metadata(ss, vs_metadata)

    return solve(model, merged_stats, solver; decompose=decompose, fix_rho=fix_rho, seed=seed, verbose=verbose)
end

"""
    fit_mle(::Type{M}, df::DataFrame; outcome=:y, firm_id=:firm_id, worker_id=:worker_id,
            solver=ExactCholesky(), rho_limit=0.99, weighting=Weighting(),
            model_adjacency=:binary, max_degree=nothing, standardize=true,
            on_missing=:drop, decompose=nothing, fix_rho=nothing, seed=42,
            strict_forest=false, verbose=false)

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
    decompose::Union{Nothing,Int}=nothing,
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

"""
    fit_mle(model::AbstractBipartiteModel, ss::BipartiteGMRFStats;
            solver=ExactCholesky(), decompose=nothing, fix_rho=nothing,
            seed=42, verbose=false)

Fit a prebuilt bipartite model (e.g. one constructed with a custom `rho_limit`
or adjacency) by maximum likelihood. Equivalent to `solve(model, ss, solver; ...)`.
"""
function fit_mle(
    model::AbstractBipartiteModel,
    ss::BipartiteGMRFStats;
    solver::AbstractGMRFSolver=ExactCholesky(),
    decompose::Union{Nothing,Int}=nothing,
    fix_rho::Union{Nothing,Float64}=nothing,
    seed::Int=42,
    verbose::Bool=false,
)
    return solve(model, ss, solver; decompose=decompose, fix_rho=fix_rho, seed=seed, verbose=verbose)
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

# ─── Helper to merge VS metadata into stats ──────────────────────────────

function _with_merged_metadata(ss::BipartiteGMRFStats, extra::NamedTuple)
    return BipartiteGMRFStats(
        ss.VtV, ss.projected_y, ss.ydot, ss.A_obs, ss.At_obs, ss.cnt_f, ss.cnt_w,
        ss.A_prior, ss.base_f_rows, ss.base_w_cols, ss.base_y, ss.base_T,
        ss.decomp_f_rows, ss.decomp_w_cols, ss.decomp_y, ss.decomp_T,
        ss.firm_ids, ss.worker_ids, ss.firm_to_index, ss.worker_to_index,
        ss.N_firms, ss.N_workers, ss.K, ss.personyear_rows,
        ss.y_mean, ss.y_std, ss.standardize,
        ss.weighting, ss.rho_eps_likelihood, ss.within_ss, ss.within_df,
        ss.personyear_within_ss, ss.log_weight_sum, ss.effective_weight_sum,
        ss.effective_weight_over_T_sum, ss.mean_effective_weight, ss.max_effective_weight,
        merge(ss.metadata, extra),
    )
end
