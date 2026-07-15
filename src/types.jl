"""
Abstract supertype for prior precision models used by `GMRFProblem`.

Concrete priors define how the bipartite adjacency matrix enters the latent
firm/worker precision matrix. Use one of `NormalizedPrior`,
`UnnormalizedPrior`, `SpectralPrior`, or `VarianceStablePrior`.
"""
abstract type AbstractGMRFPrior end

"""
Abstract supertype for marginal-likelihood solvers.

Concrete solvers control how log determinants, quadratic forms, and optimizer
steps are evaluated. Use `ExactCholesky()` for sparse direct factorization or
`HutchSLQ()` for matrix-free stochastic evaluation.
"""
abstract type AbstractGMRFSolver end

function validate_rho_limit(rho_limit::Real)
    limit = Float64(rho_limit)
    0.0 < limit < 1.0 ||
        throw(ArgumentError("rho_limit must satisfy 0 < rho_limit < 1; got $(rho_limit)."))
    return limit
end

"""
    NormalizedPrior(; adjacency=:degree, prior_adjacency=:binary, rho_limit=0.99)

Degree-normalized bipartite GMRF prior.

`prior_adjacency` controls whether the prior graph is binary (`:binary`) or
uses observed edge counts (`:counts`). This prior supports both
`ExactCholesky()` and `HutchSLQ()`. `rho_limit` is the open bound used to
transform and validate the latent firm-worker dependence parameter.
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

Unnormalized `D - rho*A` precision model.

This prior matches the paper-style degree matrix scaling and supports both
`ExactCholesky()` and `HutchSLQ()`. `rho_limit` is the open bound used to
transform and validate the latent firm-worker dependence parameter.
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

Spectral-normalized adjacency prior.

The bipartite adjacency is scaled by its leading singular value before entering
the precision matrix. `seed` controls the power-iteration initialization used
for spectral normalization. This prior currently supports `HutchSLQ()` only.
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

Variance-stable precision model for bipartite graphs.

This prior is intended for forest-like graphs where its marginal-variance
property applies. It currently supports raw observation weighting only.
`ExactCholesky()` gives deterministic likelihoods when sparse fill-in is
manageable; `HutchSLQ()` uses a congruence-scaled, common-probe estimate of the
log-determinant ratio for numerical stability. Cyclic graphs warn by default;
set `strict_forest=true` to throw `ArgumentError` instead. The default numeric
`rho_limit=0.99` is unchanged. Pass `rho_limit=:auto` to resolve an opt-in
non-backtracking feasibility limit while preparing the problem.
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

"""
    Weighting(; observations=:raw, rho_eps=nothing, target=:estimation)

Configure how repeated firm-worker observations enter the likelihood and
variance decompositions.

`observations` is one of `:raw`, `:edge`, or `:effective`. Effective weighting
requires `rho_eps` as a number in `[0, 1)` or `:estimate`. `target` controls the
default decomposition target and accepts `:estimation`, `:personyear`, or
`:edge`.
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

"""
    GMRFProblem(df; outcome=:y, firm_id=:firm_id, worker_id=:worker_id, ...)

Prepared bipartite-GMRF estimation problem.

The constructor validates and standardizes an input `DataFrame`, maps firm and
worker IDs to graph indices, builds sparse observation/prior matrices, and
stores metadata used by `solve`. Build a `GMRFProblem` once when comparing
multiple solvers on the same data.
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

const GMRFPROBLEM_LEGACY_ALIASES = Dict{Symbol,Symbol}(
    :N_F => :N_firms,
    :N_M => :N_workers,
    :firms => :firm_ids,
    :people => :worker_ids,
    :A_fm => :A_prior,
    :At_fm => :At_prior,
    :cnt_m => :cnt_w,
)

Base.propertynames(p::GMRFProblem, private::Bool=false) = fieldnames(GMRFProblem)

Base.getproperty(p::GMRFProblem, name::Symbol) = getproperty(p, Val(name))

function legacy_gmrf_property(p::GMRFProblem, legacy::Symbol, canonical::Symbol)
    Base.depwarn(
        "`GMRFProblem.$(legacy)` is deprecated; use `GMRFProblem.$(canonical)` instead.",
        :getproperty,
    )
    return getfield(p, canonical)
end

Base.getproperty(p::GMRFProblem, ::Val{:N_F}) = legacy_gmrf_property(p, :N_F, :N_firms)
Base.getproperty(p::GMRFProblem, ::Val{:N_M}) = legacy_gmrf_property(p, :N_M, :N_workers)
Base.getproperty(p::GMRFProblem, ::Val{:firms}) = legacy_gmrf_property(p, :firms, :firm_ids)
Base.getproperty(p::GMRFProblem, ::Val{:people}) = legacy_gmrf_property(p, :people, :worker_ids)
Base.getproperty(p::GMRFProblem, ::Val{:A_fm}) = legacy_gmrf_property(p, :A_fm, :A_prior)
Base.getproperty(p::GMRFProblem, ::Val{:At_fm}) = legacy_gmrf_property(p, :At_fm, :At_prior)
Base.getproperty(p::GMRFProblem, ::Val{:cnt_m}) = legacy_gmrf_property(p, :cnt_m, :cnt_w)
Base.getproperty(p::GMRFProblem, ::Val{name}) where {name} = getfield(p, name)

"""
    HutchSLQ(; logdet_probes=30, lanczos_iters=30, cg_tol=1e-6,
             cg_maxiter=700, optim_iters=1000, g_reltol=1e-7)

Matrix-free stochastic solver using Hutchinson stochastic Lanczos quadrature
for log determinants and preconditioned conjugate gradients for linear solves.

Use this solver when sparse direct factorization is too memory intensive.
Pass `seed` to `solve` or `gmrf_mle` for reproducible stochastic paths.

`g_reltol` is a *relative* Nelder-Mead convergence tolerance: the optimizer
stops once the simplex-objective spread falls below `g_reltol * max(1, |nll₀|)`,
where `nll₀` is the objective at the starting point. A relative tolerance is
scale-invariant across problem sizes, which matters here because the SLQ/PCG
objective is accurate only to a relative level (set by `cg_tol` and the
stochastic log-determinant). An absolute tolerance tight enough for a small
graph is unreachable on a large one and forces the optimizer to exhaust
`optim_iters` without converging.
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

This solver is suitable for small and medium graphs where CHOLMOD fill-in is
manageable. When `polish=true`, a finite-difference L-BFGS polishing pass is
attempted after the initial Nelder-Mead optimization.

`g_reltol` scales the Nelder-Mead convergence tolerance by the objective
magnitude at the start point, with a `1e-3` absolute floor:
`g_tol = max(1e-3, g_reltol * max(1, |nll0|))`. The floor keeps small-problem
behaviour fixed; the relative term loosens the tolerance on large problems
(where an absolute `1e-3` on the simplex-objective spread is unreachable).
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

"""
Variance decomposition returned by `prior_decomposition` or
`posterior_decomposition`.

Fields report firm, worker, cross, residual, and total variance components,
along with probe counts, decomposition target, method, convergence metadata,
and additional residual-model details.
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

"""
Fitted bipartite-GMRF model returned by `solve` and `gmrf_mle`.

Stores fitted parameters in original outcome units, optimization diagnostics,
optional prior/posterior decompositions, the prepared `GMRFProblem`, and solver
metadata.
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
    prior_decomposition::Union{VarianceDecomposition,Nothing}
    posterior_decomposition::Union{VarianceDecomposition,Nothing}
    problem::P
    prior::R
    solver::S
    theta_unconstrained::Vector{Float64}
    metadata::NamedTuple
end

"""
Cached covariance factorization returned by `prior_covariance` or
`posterior_covariance`.

Pass a `CovarianceOperator` to `cov_block` to extract covariance blocks for
selected firm and worker IDs.
"""
struct CovarianceOperator{F,R<:GMRFResult}
    kind::Symbol
    factor::F
    result::R
    units::Symbol
end

"""
Covariance block returned by `cov_block`.

`matrix` contains the extracted covariance values, while `rows` and `cols`
record entity metadata as `(side=:firm/:worker, id=...)` named tuples.
"""
struct CovarianceBlock
    matrix::Matrix{Float64}
    rows::Vector{Any}
    cols::Vector{Any}
    kind::Symbol
    units::Symbol
end
