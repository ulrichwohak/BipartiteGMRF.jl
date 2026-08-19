# ═══════════════════════════════════════════════════════════════════════════
# fit_mle: standard MLE interface (Distributions.jl convention)
# ═══════════════════════════════════════════════════════════════════════════

"""
    fit_mle(::Type{M}, ss::BipartiteGMRFStats; solver=ExactCholesky(),
            rho_limit=0.99, fix_rho=nothing, init=nothing, seed=42,
            strict_forest=false, verbose=false)

Fit a bipartite GMRF model by maximum likelihood using precomputed sufficient
statistics. Returns a [`GMRFResult`](@ref). This extends `Distributions.fit_mle`
and follows its `fit_mle(D, suffstats)` pattern.

`rho_limit=:auto` (variance-stable model only) resolves the limit from the
non-backtracking spectrum. Use [`decompose`](@ref) on the result for variance
decompositions.

# Warm starts (`init`)

`init` is a `NamedTuple` of starting values, every field optional; whatever is
missing keeps the built-in heuristic (`rho = min(0.5, 0.5 rho_limit)`,
`sigma_a = 0.7`, `sigma_z = 0.04`, `sigma_epsilon = 0.4`), which assumes a
unit-variance outcome and is not tuned per dataset.

```julia
r = fit_mle(BipartiteVarianceStableModel, f, w, y)
r2 = fit_mle(BipartiteVarianceStableModel, f, w, y; init = params(r))
```

Fields: `rho`, `sigma_a`, `sigma_z`, `sigma_epsilon`, plus `rho_eps` (effective
weighting, default 0.5), `eta` (`error_eta = :estimate`, default 0.5), `omega`
(`error_groups`, default 1 per class), and `phi` / `r` / `delta`
(`error_blocks = :iw`). `sigma_epsilon` is not an EMIW coordinate: it converts
to the error scale through `omega_bar = phi*delta/(delta-2)`, and is ignored
with a warning if `phi` is also given.

A field for a parameter this fit does not estimate is an error. A field for a
parameter that is present but held fixed — `fix_rho`, a numeric `error_eta`, a
numeric `rho_eps` under effective weighting, or a solver-fixed `EMIWBlocks(r=)`
/ `(delta=)` — is ignored, with a warning unless it already agrees with the
pinned value. A `nothing`-valued field counts as not supplied, so
`init = params(result)` works directly, including when it carries a pinned
parameter back into an identically configured fit.

**Units are the original outcome's**, matching what the result reports — the
scale parameters are divided by the outcome's standard deviation internally,
so a warm start taken from a previous `GMRFResult` or from a raw-data
minimum-distance estimate needs no rescaling. `omega` may be given either as
the free classes `2..C` or as the full ladder reported in
`error_class_variances`, whose leading entry is pinned at 1.

!!! note "A warm start is not guaranteed to be faster"
    The Nelder-Mead simplex is built by perturbing each coordinate by 50%
    (Optim's default), so it still spans a wide region around `init`. A
    coordinate sitting at exactly zero on the unconstrained scale instead gets
    an absolute step of 0.025, and one merely close to zero gets `0.5|x|`,
    which is smaller still; a warm start tends to produce exactly those
    (`log omega = 0`, `atanh(eta) ~ 0`, `log sigma ~ 0` at `sigma ~ 1`), and a
    simplex that is near-degenerate in one coordinate can satisfy the
    convergence test almost immediately. `ExactCholesky` is protected by its
    L-BFGS polish; `HutchSLQ` has no polish, so check that a warm-started
    `HutchSLQ` fit actually moved. The stopping tolerance is also scaled by the
    objective at the starting point, which under `HutchSLQ` means a better
    start can buy a *tighter* tolerance and cost extra iterations
    (`ExactCholesky` floors it, so it usually does not move). See issue #115.
"""
function fit_mle(
    ::Type{M},
    ss::BipartiteGMRFStats;
    solver::AbstractGMRFSolver=ExactCholesky(),
    rho_limit::Union{Real,Symbol}=0.99,
    fix_rho::Union{Nothing,Float64}=nothing,
    init::Union{Nothing,NamedTuple}=nothing,
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
    return solve(model, ss, solver; fix_rho=fix_rho, init=init, seed=seed, verbose=verbose)
end

"""
    fit_mle(::Type{M}, f_idx, w_idx, y; kwargs...)

One-step convenience: compute sufficient statistics from parallel observation
vectors (firm index, worker index, outcome) and fit. Accepts the keyword
arguments of both [`suffstats`](@ref) (`n_firms`, `n_workers`, `weighting`,
`model_adjacency`, `match_id`, `standardize`, `X`, `error_cov`,
`error_groups`, `error_group_cap`, `error_eta`, `edge_index`, `error_blocks`,
`firm_group`) and the suffstats-based `fit_mle` method (including `init` — see
there for the warm-start fields and their units).
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
    error_blocks::Union{Nothing,Symbol}=nothing,
    firm_group::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    solver::AbstractGMRFSolver=ExactCholesky(),
    rho_limit::Union{Real,Symbol}=0.99,
    fix_rho::Union{Nothing,Float64}=nothing,
    init::Union{Nothing,NamedTuple}=nothing,
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
        error_blocks=error_blocks,
        firm_group=firm_group,
    )
    return fit_mle(M, ss;
        solver=solver,
        rho_limit=rho_limit,
        fix_rho=fix_rho,
        init=init,
        seed=seed,
        strict_forest=strict_forest,
        verbose=verbose,
    )
end

"""
    fit_mle(model::AbstractBipartiteModel, ss::BipartiteGMRFStats;
            solver=ExactCholesky(), fix_rho=nothing, init=nothing,
            seed=42, verbose=false)

Fit a prebuilt bipartite model (e.g. one constructed with a custom `rho_limit`
or adjacency) by maximum likelihood. Equivalent to `solve(model, ss, solver; ...)`.
See the suffstats-based method for the `init` warm-start fields and their units.
"""
function fit_mle(
    model::AbstractBipartiteModel,
    ss::BipartiteGMRFStats;
    solver::AbstractGMRFSolver=ExactCholesky(),
    fix_rho::Union{Nothing,Float64}=nothing,
    init::Union{Nothing,NamedTuple}=nothing,
    seed::Int=42,
    verbose::Bool=false,
)
    return solve(model, ss, solver; fix_rho=fix_rho, init=init, seed=seed, verbose=verbose)
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
