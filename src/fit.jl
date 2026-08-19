# ═══════════════════════════════════════════════════════════════════════════
# fit_mle: standard MLE interface (Distributions.jl convention)
# ═══════════════════════════════════════════════════════════════════════════

"""
    fit_mle(::Type{M}, ss::BipartiteGMRFStats; solver=ExactCholesky(),
            rho_limit=0.99, fix_rho=nothing, seed=42,
            strict_forest=false, verbose=false)

Fit a bipartite GMRF model by maximum likelihood using precomputed sufficient
statistics. Returns a [`GMRFResult`](@ref). This extends `Distributions.fit_mle`
and follows its `fit_mle(D, suffstats)` pattern.

`rho_limit=:auto` (variance-stable model only) resolves the limit from the
non-backtracking spectrum. Use [`decompose`](@ref) on the result for variance
decompositions.
"""
function fit_mle(
    ::Type{M},
    ss::BipartiteGMRFStats;
    solver::AbstractGMRFSolver=ExactCholesky(),
    rho_limit::Union{Real,Symbol}=0.99,
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
    model = _build_model(M, ss.A_prior, rho_limit; seed=seed, strict_forest=strict_forest)
    return solve(model, ss, solver; fix_rho=fix_rho, seed=seed, verbose=verbose)
end

"""
    fit_mle(::Type{M}, f_idx, w_idx, y; kwargs...)

One-step convenience: compute sufficient statistics from parallel observation
vectors (firm index, worker index, outcome) and fit. Accepts the keyword
arguments of both [`suffstats`](@ref) (`n_firms`, `n_workers`, `weighting`,
`model_adjacency`, `match_id`, `standardize`, `X`, `error_cov`,
`error_groups`, `error_group_cap`) and the suffstats-based `fit_mle` method.
"""
function fit_mle(
    ::Type{M},
    f_idx::AbstractVector{<:Integer},
    w_idx::AbstractVector{<:Integer},
    y::AbstractVector{<:Real};
    n_firms::Integer=isempty(f_idx) ? 0 : maximum(f_idx),
    n_workers::Integer=isempty(w_idx) ? 0 : maximum(w_idx),
    weighting::Weighting=Weighting(),
    model_adjacency::Symbol=:binary,
    match_id::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    standardize::Bool=true,
    X::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    error_cov::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    error_groups::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    error_group_cap::Integer=8,
    error_eta::Union{Nothing,Real,Symbol}=nothing,
    edge_index::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    solver::AbstractGMRFSolver=ExactCholesky(),
    rho_limit::Union{Real,Symbol}=0.99,
    fix_rho::Union{Nothing,Float64}=nothing,
    seed::Int=42,
    strict_forest::Bool=false,
    verbose::Bool=false,
) where {M<:AbstractBipartiteModel}
    ss = suffstats(M, f_idx, w_idx, y;
        n_firms=n_firms,
        n_workers=n_workers,
        weighting=weighting,
        model_adjacency=model_adjacency,
        match_id=match_id,
        standardize=standardize,
        X=X,
        error_cov=error_cov,
        error_groups=error_groups,
        error_group_cap=error_group_cap,
        error_eta=error_eta,
        edge_index=edge_index,
    )
    return fit_mle(M, ss;
        solver=solver,
        rho_limit=rho_limit,
        fix_rho=fix_rho,
        seed=seed,
        strict_forest=strict_forest,
        verbose=verbose,
    )
end

"""
    fit_mle(model::AbstractBipartiteModel, ss::BipartiteGMRFStats;
            solver=ExactCholesky(), fix_rho=nothing, seed=42, verbose=false)

Fit a prebuilt bipartite model (e.g. one constructed with a custom `rho_limit`
or adjacency) by maximum likelihood. Equivalent to `solve(model, ss, solver; ...)`.
"""
function fit_mle(
    model::AbstractBipartiteModel,
    ss::BipartiteGMRFStats;
    solver::AbstractGMRFSolver=ExactCholesky(),
    fix_rho::Union{Nothing,Float64}=nothing,
    seed::Int=42,
    verbose::Bool=false,
)
    return solve(model, ss, solver; fix_rho=fix_rho, seed=seed, verbose=verbose)
end

# ─── Model construction dispatch ─────────────────────────────────────────

function _build_model(::Type{<:BipartiteNormalizedModel}, A::SparseMatrixCSC{Float64,Int}, rho_limit::Real; kwargs...)
    return BipartiteNormalizedModel(A; rho_limit=rho_limit)
end

function _build_model(::Type{<:BipartiteUnnormalizedModel}, A::SparseMatrixCSC{Float64,Int}, rho_limit::Real; kwargs...)
    return BipartiteUnnormalizedModel(A; rho_limit=rho_limit)
end

function _build_model(::Type{<:BipartiteSpectralModel}, A::SparseMatrixCSC{Float64,Int}, rho_limit::Real; seed::Int=12345, kwargs...)
    return BipartiteSpectralModel(A; rho_limit=rho_limit, seed=seed)
end

function _build_model(::Type{<:BipartiteVarianceStableModel}, A::SparseMatrixCSC{Float64,Int}, rho_limit::Union{Real,Symbol}; strict_forest::Bool=false, kwargs...)
    return BipartiteVarianceStableModel(A; strict_forest=strict_forest, rho_limit=rho_limit)
end
