# ═══════════════════════════════════════════════════════════════════════════
# Abstract bipartite model hierarchy
# ═══════════════════════════════════════════════════════════════════════════

"""
Abstract supertype for bipartite GMRF latent models.

All bipartite models share a common graph structure (rectangular adjacency
matrix, firm/worker degrees) and differ in how the precision matrix is
constructed from hyperparameters `(ρ, σ_a, σ_z)`.

Subtypes implement the GaussianMarkovRandomFields.jl `LatentModel` interface:
`precision_matrix`, `mean`, `hyperparameters`, `constraints`, `model_name`, `length`.
"""
abstract type AbstractBipartiteModel <: LatentModel end

# ═══════════════════════════════════════════════════════════════════════════
# Bipartite graph data
# ═══════════════════════════════════════════════════════════════════════════

"""
    BipartiteGraph

Stores the bipartite adjacency matrix and precomputed degree vectors
shared by all bipartite model types.
"""
struct BipartiteGraph
    A::SparseMatrixCSC{Float64,Int}        # n_firms × n_workers adjacency
    At::SparseMatrixCSC{Float64,Int}       # transpose
    d_f::Vector{Float64}                   # firm degrees
    d_w::Vector{Float64}                   # worker degrees
    n_firms::Int
    n_workers::Int
end

function BipartiteGraph(A::SparseMatrixCSC{Float64,Int})
    n_firms, n_workers = size(A)
    At = copy(transpose(A))
    d_f = vec(sum(A; dims=2))
    d_w = vec(sum(A; dims=1))
    any(d_f .<= 0) && throw(ArgumentError("Zero-degree firm node detected."))
    any(d_w .<= 0) && throw(ArgumentError("Zero-degree worker node detected."))
    return BipartiteGraph(A, At, Float64.(d_f), Float64.(d_w), n_firms, n_workers)
end

# ═══════════════════════════════════════════════════════════════════════════
# Model specification types (lightweight, no graph data)
#
# These are the user-facing API for specifying which model to use. They
# are passed to gmrf_mle() / GMRFProblem() and resolved into full
# AbstractBipartiteModel subtypes by prepare.jl once the adjacency is known.
# ═══════════════════════════════════════════════════════════════════════════

"""
Abstract supertype for model specifications passed to `gmrf_mle` / `GMRFProblem`.
"""
abstract type AbstractGMRFPrior end

function validate_rho_limit(rho_limit::Real)
    limit = Float64(rho_limit)
    0.0 < limit < 1.0 ||
        throw(ArgumentError("rho_limit must satisfy 0 < rho_limit < 1; got $(rho_limit)."))
    return limit
end

"""
    NormalizedPrior(; adjacency=:degree, prior_adjacency=:binary, rho_limit=0.99)

Degree-normalized bipartite GMRF model specification.
"""
struct NormalizedPrior <: AbstractGMRFPrior
    adjacency::Symbol
    prior_adjacency::Symbol
    rho_limit::Float64
    function NormalizedPrior(;
        adjacency::Symbol=:degree,
        prior_adjacency::Symbol=:binary,
        rho_limit::Real=0.99,
    )
        adjacency == :degree ||
            throw(ArgumentError("NormalizedPrior only supports adjacency=:degree; got $(adjacency)."))
        prior_adjacency in (:binary, :counts) ||
            throw(ArgumentError("prior_adjacency must be :binary or :counts; got $(prior_adjacency)."))
        new(adjacency, prior_adjacency, validate_rho_limit(rho_limit))
    end
end

"""
    UnnormalizedPrior(; prior_adjacency=:binary, rho_limit=0.99)

Unnormalized `D - ρA` model specification.
"""
struct UnnormalizedPrior <: AbstractGMRFPrior
    prior_adjacency::Symbol
    rho_limit::Float64
    function UnnormalizedPrior(; prior_adjacency::Symbol=:binary, rho_limit::Real=0.99)
        prior_adjacency in (:binary, :counts) ||
            throw(ArgumentError("prior_adjacency must be :binary or :counts; got $(prior_adjacency)."))
        new(prior_adjacency, validate_rho_limit(rho_limit))
    end
end

"""
    SpectralPrior(; prior_adjacency=:binary, seed=12345, rho_limit=0.99)

Spectral-normalized model specification.
"""
struct SpectralPrior <: AbstractGMRFPrior
    prior_adjacency::Symbol
    seed::Int
    rho_limit::Float64
    function SpectralPrior(;
        prior_adjacency::Symbol=:binary,
        seed::Int=12345,
        rho_limit::Real=0.99,
    )
        prior_adjacency in (:binary, :counts) ||
            throw(ArgumentError("prior_adjacency must be :binary or :counts; got $(prior_adjacency)."))
        new(prior_adjacency, seed, validate_rho_limit(rho_limit))
    end
end

"""
    VarianceStablePrior(; strict_forest=false, rho_limit=0.99)

Variance-stable model specification for forest-like graphs.
"""
struct VarianceStablePrior{L<:Union{Float64,Symbol}} <: AbstractGMRFPrior
    strict_forest::Bool
    rho_limit::L
    function VarianceStablePrior(strict_forest::Bool, rho_limit::Float64)
        return new{Float64}(strict_forest, validate_rho_limit(rho_limit))
    end
    function VarianceStablePrior(strict_forest::Bool, rho_limit::Symbol)
        rho_limit == :auto ||
            throw(ArgumentError("VarianceStablePrior rho_limit symbol must be :auto; got $(rho_limit)."))
        return new{Symbol}(strict_forest, rho_limit)
    end
end

function VarianceStablePrior(;
    strict_forest::Bool=false,
    rho_limit::Union{Real,Symbol}=0.99,
)
    if rho_limit isa Symbol
        rho_limit == :auto ||
            throw(ArgumentError("VarianceStablePrior rho_limit symbol must be :auto; got $(rho_limit)."))
        return VarianceStablePrior(strict_forest, rho_limit)
    end
    return VarianceStablePrior(strict_forest, validate_rho_limit(rho_limit))
end

rho_limit(prior::AbstractGMRFPrior) = prior.rho_limit
rho_limit(::VarianceStablePrior{Symbol}) =
    throw(ArgumentError("rho_limit=:auto must be resolved by constructing a GMRFProblem."))

# ═══════════════════════════════════════════════════════════════════════════
# Concrete bipartite model types (graph-bound, implement LatentModel)
# ═══════════════════════════════════════════════════════════════════════════

"""
Degree-normalized bipartite GMRF. Implements `GaussianMarkovRandomFields.LatentModel`.
"""
struct BipartiteNormalizedModel{Alg} <: AbstractBipartiteModel
    graph::BipartiteGraph
    df_is::Vector{Float64}    # D_f^{-1/2}
    dw_is::Vector{Float64}    # D_w^{-1/2}
    rho_limit::Float64
    alg::Alg
end

function BipartiteNormalizedModel(
    A::SparseMatrixCSC{Float64,Int};
    rho_limit::Real=0.99,
    alg=CHOLMODFactorization(),
)
    g = BipartiteGraph(A)
    df_is = 1.0 ./ sqrt.(g.d_f)
    dw_is = 1.0 ./ sqrt.(g.d_w)
    return BipartiteNormalizedModel(g, df_is, dw_is, validate_rho_limit(rho_limit), alg)
end

"""
Unnormalized `D - ρA` bipartite GMRF. Implements `GaussianMarkovRandomFields.LatentModel`.
"""
struct BipartiteUnnormalizedModel{Alg} <: AbstractBipartiteModel
    graph::BipartiteGraph
    rho_limit::Float64
    alg::Alg
end

function BipartiteUnnormalizedModel(
    A::SparseMatrixCSC{Float64,Int};
    rho_limit::Real=0.99,
    alg=CHOLMODFactorization(),
)
    g = BipartiteGraph(A)
    return BipartiteUnnormalizedModel(g, validate_rho_limit(rho_limit), alg)
end

"""
Spectral-normalized bipartite GMRF. Implements `GaussianMarkovRandomFields.LatentModel`.
"""
struct BipartiteSpectralModel{Alg} <: AbstractBipartiteModel
    graph::BipartiteGraph
    spectral_is::Float64   # 1/√s₁
    rho_limit::Float64
    alg::Alg
end

function BipartiteSpectralModel(
    A::SparseMatrixCSC{Float64,Int};
    rho_limit::Real=0.99,
    seed::Int=12345,
    alg=CHOLMODFactorization(),
)
    g = BipartiteGraph(A)
    s1 = leading_singular_value(g.A, g.At; seed=seed)
    s1 > 0 || throw(ArgumentError("Cannot spectral-normalize an empty adjacency matrix."))
    return BipartiteSpectralModel(g, 1.0 / sqrt(s1), validate_rho_limit(rho_limit), alg)
end

"""
Variance-stable bipartite GMRF for forest-like graphs.
Implements `GaussianMarkovRandomFields.LatentModel`.
"""
struct BipartiteVarianceStableModel{L<:Union{Float64,Symbol},Alg} <: AbstractBipartiteModel
    graph::BipartiteGraph
    strict_forest::Bool
    rho_limit::L
    alg::Alg
end

function BipartiteVarianceStableModel(
    A::SparseMatrixCSC{Float64,Int};
    strict_forest::Bool=false,
    rho_limit::Union{Real,Symbol}=0.99,
    alg=CHOLMODFactorization(),
)
    g = BipartiteGraph(_binarize(A))

    if !is_forest(g.A)
        msg = "Input graph contains a cycle; variance-stable model no longer guarantees degree-independent marginal variances."
        strict_forest && throw(ArgumentError(msg))
        @warn msg
    end

    if rho_limit isa Symbol
        rho_limit == :auto ||
            throw(ArgumentError("rho_limit symbol must be :auto; got $(rho_limit)."))
        return BipartiteVarianceStableModel(g, strict_forest, rho_limit, alg)
    end
    return BipartiteVarianceStableModel(g, strict_forest, validate_rho_limit(rho_limit), alg)
end

# ─── rho_limit for model types ────────────────────────────────────────────
rho_limit(m::AbstractBipartiteModel) = m.rho_limit
rho_limit(::BipartiteVarianceStableModel{Symbol}) =
    throw(ArgumentError("rho_limit=:auto must be resolved before use."))

# ─── GaussianMarkovRandomFields.jl LatentModel interface ──────────────────

Base.length(m::AbstractBipartiteModel) = m.graph.n_firms + m.graph.n_workers

GaussianMarkovRandomFields.hyperparameters(::AbstractBipartiteModel) = (ρ = Real, σ_a = Real, σ_z = Real)
GaussianMarkovRandomFields.constraints(::AbstractBipartiteModel; kwargs...) = nothing

GaussianMarkovRandomFields.model_name(::BipartiteNormalizedModel) = :bipartite_normalized
GaussianMarkovRandomFields.model_name(::BipartiteUnnormalizedModel) = :bipartite_unnormalized
GaussianMarkovRandomFields.model_name(::BipartiteSpectralModel) = :bipartite_spectral
GaussianMarkovRandomFields.model_name(::BipartiteVarianceStableModel) = :bipartite_variance_stable

GaussianMarkovRandomFields.mean(m::AbstractBipartiteModel; kwargs...) = zeros(length(m))

# ─── precision_matrix implementations ──────────────────────────────────────

function GaussianMarkovRandomFields.precision_matrix(
    m::BipartiteNormalizedModel;
    ρ::Real, σ_a::Real, σ_z::Real, kwargs...,
)
    _validate_bipartite_params(ρ, σ_a, σ_z, m.rho_limit)
    inv_sa2 = 1.0 / σ_a^2
    inv_sz2 = 1.0 / σ_z^2
    cross = ρ / (σ_a * σ_z)
    g = m.graph
    W = spdiagm(0 => m.df_is) * g.A * spdiagm(0 => m.dw_is)
    Wt = copy(transpose(W))
    return [
        spdiagm(0 => fill(inv_sa2, g.n_firms))   (-cross .* W)
        (-cross .* Wt)                             spdiagm(0 => fill(inv_sz2, g.n_workers))
    ]
end

function GaussianMarkovRandomFields.precision_matrix(
    m::BipartiteUnnormalizedModel;
    ρ::Real, σ_a::Real, σ_z::Real, kwargs...,
)
    _validate_bipartite_params(ρ, σ_a, σ_z, m.rho_limit)
    inv_sa2 = 1.0 / σ_a^2
    inv_sz2 = 1.0 / σ_z^2
    cross = ρ / (σ_a * σ_z)
    g = m.graph
    return [
        spdiagm(0 => g.d_f .* inv_sa2)   (-cross .* g.A)
        (-cross .* g.At)                   spdiagm(0 => g.d_w .* inv_sz2)
    ]
end

function GaussianMarkovRandomFields.precision_matrix(
    m::BipartiteSpectralModel;
    ρ::Real, σ_a::Real, σ_z::Real, kwargs...,
)
    _validate_bipartite_params(ρ, σ_a, σ_z, m.rho_limit)
    inv_sa2 = 1.0 / σ_a^2
    inv_sz2 = 1.0 / σ_z^2
    cross = ρ / (σ_a * σ_z)
    g = m.graph
    s = m.spectral_is  # 1/√s₁
    W = (s * s) .* g.A   # A / s₁
    Wt = copy(transpose(W))
    return [
        spdiagm(0 => fill(inv_sa2, g.n_firms))   (-cross .* W)
        (-cross .* Wt)                             spdiagm(0 => fill(inv_sz2, g.n_workers))
    ]
end

function GaussianMarkovRandomFields.precision_matrix(
    m::BipartiteVarianceStableModel{Float64};
    ρ::Real, σ_a::Real, σ_z::Real, kwargs...,
)
    _validate_bipartite_params(ρ, σ_a, σ_z, m.rho_limit)
    inv_sa2 = 1.0 / σ_a^2
    inv_sz2 = 1.0 / σ_z^2
    cross = ρ / (σ_a * σ_z)
    rho_sq = ρ^2
    g = m.graph
    diag_f = (1.0 .+ rho_sq .* (g.d_f .- 1.0)) .* inv_sa2
    diag_w = (1.0 .+ rho_sq .* (g.d_w .- 1.0)) .* inv_sz2
    return [
        spdiagm(0 => diag_f)   (-cross .* g.A)
        (-cross .* g.At)        spdiagm(0 => diag_w)
    ]
end

# ─── Helpers ───────────────────────────────────────────────────────────────

function _validate_bipartite_params(ρ::Real, σ_a::Real, σ_z::Real, rho_limit::Float64)
    abs(ρ) < rho_limit || throw(ArgumentError("ρ must satisfy |ρ| < $(rho_limit); got $(ρ)."))
    σ_a > 0 || throw(ArgumentError("σ_a must be positive; got $(σ_a)."))
    σ_z > 0 || throw(ArgumentError("σ_z must be positive; got $(σ_z)."))
end

function _binarize(A::SparseMatrixCSC{Float64,Int})
    B = copy(A)
    B.nzval .= 1.0
    return B
end

# ═══════════════════════════════════════════════════════════════════════════
# Observation weighting
# ═══════════════════════════════════════════════════════════════════════════

"""
    Weighting(; observations=:raw, rho_eps=nothing, target=:estimation)

Configure how repeated firm-worker observations enter the likelihood and
variance decompositions.
"""
struct Weighting
    observations::Symbol
    rho_eps::Union{Nothing,Float64,Symbol}
    target::Symbol
    function Weighting(;
        observations::Symbol=:raw,
        rho_eps::Union{Nothing,Float64,Symbol}=nothing,
        target::Symbol=:estimation,
    )
        observations in (:raw, :edge, :effective) ||
            throw(ArgumentError("observations must be :raw, :edge, or :effective; got $(observations)."))
        target = normalize_decomp_target(target)
        if observations == :effective
            rho_eps === nothing &&
                throw(ArgumentError("effective weighting requires rho_eps as a Float64 or :estimate."))
            if rho_eps isa Symbol
                rho_eps == :estimate ||
                    throw(ArgumentError("rho_eps symbol must be :estimate; got $(rho_eps)."))
            elseif !(0.0 <= rho_eps < 1.0)
                throw(ArgumentError("rho_eps must satisfy 0 <= rho_eps < 1; got $(rho_eps)."))
            end
        elseif rho_eps !== nothing
            throw(ArgumentError("rho_eps is only meaningful with observations=:effective."))
        end
        new(observations, rho_eps, target)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Solver types
# ═══════════════════════════════════════════════════════════════════════════

"""
Abstract supertype for marginal-likelihood solvers.
"""
abstract type AbstractGMRFSolver end

"""
    HutchSLQ(; logdet_probes=30, lanczos_iters=30, cg_tol=1e-6,
             cg_maxiter=700, optim_iters=1000, g_reltol=1e-7)

Matrix-free stochastic solver using Hutchinson stochastic Lanczos quadrature
for log determinants and preconditioned conjugate gradients for linear solves.
"""
struct HutchSLQ <: AbstractGMRFSolver
    logdet_probes::Int
    lanczos_iters::Int
    cg_tol::Float64
    cg_maxiter::Int
    optim_iters::Int
    g_reltol::Float64
    function HutchSLQ(;
        logdet_probes::Int=30,
        lanczos_iters::Int=30,
        cg_tol::Float64=1e-6,
        cg_maxiter::Int=700,
        optim_iters::Int=1000,
        g_reltol::Float64=1e-7,
    )
        logdet_probes > 0 || throw(ArgumentError("logdet_probes must be positive."))
        lanczos_iters > 0 || throw(ArgumentError("lanczos_iters must be positive."))
        cg_tol > 0 || throw(ArgumentError("cg_tol must be positive."))
        cg_maxiter > 0 || throw(ArgumentError("cg_maxiter must be positive."))
        optim_iters > 0 || throw(ArgumentError("optim_iters must be positive."))
        g_reltol > 0 || throw(ArgumentError("g_reltol must be positive."))
        new(logdet_probes, lanczos_iters, cg_tol, cg_maxiter, optim_iters, g_reltol)
    end
end

"""
    ExactCholesky(; optim_iters=200, polish=true, autodiff=:finitediff, g_reltol=1e-7)

Deterministic solver based on sparse Cholesky factorizations.
"""
struct ExactCholesky <: AbstractGMRFSolver
    optim_iters::Int
    polish::Bool
    autodiff::Symbol
    g_reltol::Float64
    function ExactCholesky(; optim_iters::Int=200, polish::Bool=true,
                           autodiff::Symbol=:finitediff, g_reltol::Float64=1e-7)
        optim_iters > 0 || throw(ArgumentError("optim_iters must be positive."))
        autodiff in (:finitediff, :none) ||
            throw(ArgumentError("ExactCholesky supports autodiff=:finitediff or :none; got $(autodiff)."))
        g_reltol > 0 || throw(ArgumentError("g_reltol must be positive."))
        new(optim_iters, polish, autodiff, g_reltol)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Prepared estimation problem
# ═══════════════════════════════════════════════════════════════════════════

"""
    GMRFProblem(df; outcome=:y, firm_id=:firm_id, worker_id=:worker_id, ...)

Prepared bipartite-GMRF estimation problem.

The constructor validates and standardizes an input `DataFrame`, maps firm and
worker IDs to graph indices, builds sparse observation/prior matrices, and
stores metadata used by `solve`.
"""
struct GMRFProblem{F,W}
    y::Vector{Float64}
    ydot::Float64
    projected_y::Vector{Float64}
    VtV::SparseMatrixCSC{Float64,Int}
    A_prior::SparseMatrixCSC{Float64,Int}
    At_prior::SparseMatrixCSC{Float64,Int}
    A_obs::SparseMatrixCSC{Float64,Int}
    At_obs::SparseMatrixCSC{Float64,Int}
    d_f::Vector{Float64}
    d_w::Vector{Float64}
    cnt_f::Vector{Float64}
    cnt_w::Vector{Float64}
    df_is::Vector{Float64}
    dw_is::Vector{Float64}
    diag_f::Vector{Float64}
    diag_w::Vector{Float64}
    firm_ids::Vector{F}
    worker_ids::Vector{W}
    firm_to_index::Dict{F,Int}
    worker_to_index::Dict{W,Int}
    base_f_rows::Vector{Int}
    base_w_cols::Vector{Int}
    base_y::Vector{Float64}
    base_T::Vector{Int}
    decomp_f_rows::Vector{Int}
    decomp_w_cols::Vector{Int}
    decomp_y::Vector{Float64}
    decomp_T::Vector{Int}
    N_firms::Int
    N_workers::Int
    K::Int
    personyear_rows::Int
    y_mean::Float64
    y_std::Float64
    standardize::Bool
    prior::AbstractGMRFPrior
    model::AbstractBipartiteModel
    weighting::Weighting
    rho_eps_likelihood::Union{Nothing,Float64}
    within_ss::Float64
    within_df::Int
    personyear_within_ss::Float64
    log_weight_sum::Float64
    effective_weight_sum::Float64
    effective_weight_over_T_sum::Float64
    mean_effective_weight::Float64
    max_effective_weight::Float64
    metadata::NamedTuple
end

# ═══════════════════════════════════════════════════════════════════════════
# Variance decomposition
# ═══════════════════════════════════════════════════════════════════════════

"""
Variance decomposition returned by `decompose(result; kind=:model)` or
`decompose(result; kind=:fitted)`.
"""
struct VarianceDecomposition
    V_firm::Float64
    V_worker::Float64
    V_cross::Float64
    V_epsilon::Float64
    V_total::Float64
    n_probes::Int
    target::Symbol
    kind::Symbol
    method::Symbol
    pcg_converged::Union{Nothing,Int}
    metadata::NamedTuple
end

# ═══════════════════════════════════════════════════════════════════════════
# Estimation result
# ═══════════════════════════════════════════════════════════════════════════

"""
Fitted bipartite-GMRF model returned by `solve` and `gmrf_mle`.
"""
struct GMRFResult{
    P<:GMRFProblem,
    R<:AbstractGMRFPrior,
    S<:AbstractGMRFSolver,
}
    rho::Float64
    sigma_a::Float64
    sigma_z::Float64
    sigma_epsilon::Float64
    rho_eps::Union{Nothing,Float64}
    nll::Float64
    converged::Bool
    iterations::Int
    obj_evals::Int
    optimization_time::Float64
    model_decomposition::Union{VarianceDecomposition,Nothing}
    fitted_decomposition::Union{VarianceDecomposition,Nothing}
    problem::P
    prior::R
    solver::S
    theta_unconstrained::Vector{Float64}
    metadata::NamedTuple
end

# ═══════════════════════════════════════════════════════════════════════════
# Covariance types
# ═══════════════════════════════════════════════════════════════════════════

"""
Cached covariance factorization returned by `covariance(result; kind=:model)` or
`covariance(result; kind=:fitted)`.
"""
struct CovarianceOperator{F,R<:GMRFResult}
    kind::Symbol
    factor::F
    result::R
    units::Symbol
end

"""
Covariance block returned by `cov_block`.
"""
struct CovarianceBlock
    matrix::Matrix{Float64}
    rows::Vector{Any}
    cols::Vector{Any}
    kind::Symbol
    units::Symbol
end

# ═══════════════════════════════════════════════════════════════════════════
# Convenience: construct a LatentModel from a spec + adjacency
# ═══════════════════════════════════════════════════════════════════════════

"""
    to_model(spec::AbstractGMRFPrior, A_prior) -> AbstractBipartiteModel

Convert a lightweight model specification into a full `LatentModel` subtype
bound to the given adjacency matrix.
"""
function to_model(spec::NormalizedPrior, A::SparseMatrixCSC{Float64,Int})
    return BipartiteNormalizedModel(A; rho_limit=spec.rho_limit)
end

function to_model(spec::UnnormalizedPrior, A::SparseMatrixCSC{Float64,Int})
    return BipartiteUnnormalizedModel(A; rho_limit=spec.rho_limit)
end

function to_model(spec::SpectralPrior, A::SparseMatrixCSC{Float64,Int})
    return BipartiteSpectralModel(A; rho_limit=spec.rho_limit, seed=spec.seed)
end

function to_model(spec::VarianceStablePrior, A::SparseMatrixCSC{Float64,Int})
    return BipartiteVarianceStableModel(A;
        strict_forest=spec.strict_forest,
        rho_limit=spec.rho_limit,
    )
end
