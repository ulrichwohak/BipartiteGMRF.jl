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

"""
    NormalizedPrior(; adjacency=:degree, prior_adjacency=:binary)

Degree-normalized bipartite GMRF prior.

`prior_adjacency` controls whether the prior graph is binary (`:binary`) or
uses observed edge counts (`:counts`). This prior supports both
`ExactCholesky()` and `HutchSLQ()`.
"""
struct NormalizedPrior <: AbstractGMRFPrior
    adjacency::Symbol
    prior_adjacency::Symbol
    function NormalizedPrior(; adjacency::Symbol=:degree, prior_adjacency::Symbol=:binary)
        adjacency == :degree ||
            throw(ArgumentError("NormalizedPrior only supports adjacency=:degree; got $(adjacency)."))
        prior_adjacency in (:binary, :counts) ||
            throw(ArgumentError("prior_adjacency must be :binary or :counts; got $(prior_adjacency)."))
        new(adjacency, prior_adjacency)
    end
end

"""
    UnnormalizedPrior(; prior_adjacency=:binary)

Unnormalized `D - rho*A` precision model.

This prior matches the paper-style degree matrix scaling and supports both
`ExactCholesky()` and `HutchSLQ()`.
"""
struct UnnormalizedPrior <: AbstractGMRFPrior
    prior_adjacency::Symbol
    function UnnormalizedPrior(; prior_adjacency::Symbol=:binary)
        prior_adjacency in (:binary, :counts) ||
            throw(ArgumentError("prior_adjacency must be :binary or :counts; got $(prior_adjacency)."))
        new(prior_adjacency)
    end
end

"""
    SpectralPrior(; prior_adjacency=:binary)

Spectral-normalized adjacency prior.

The bipartite adjacency is scaled by its leading singular value before entering
the precision matrix. This prior currently supports `HutchSLQ()` only.
"""
struct SpectralPrior <: AbstractGMRFPrior
    prior_adjacency::Symbol
    seed::Int
    function SpectralPrior(; prior_adjacency::Symbol=:binary, seed::Int=12345)
        prior_adjacency in (:binary, :counts) ||
            throw(ArgumentError("prior_adjacency must be :binary or :counts; got $(prior_adjacency)."))
        new(prior_adjacency, seed)
    end
end

"""
    VarianceStablePrior()

Variance-stable precision model for bipartite graphs.

This prior is intended for forest-like graphs where its marginal-variance
property applies. It currently supports raw observation weighting and
`HutchSLQ()` only.
"""
struct VarianceStablePrior <: AbstractGMRFPrior end

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
struct GMRFProblem
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
    firm_ids::Vector{Any}
    worker_ids::Vector{Any}
    firm_to_index::Dict{Any,Int}
    worker_to_index::Dict{Any,Int}
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

Base.propertynames(p::GMRFProblem, private::Bool=false) = (
    fieldnames(GMRFProblem)...,
    :N_F,
    :N_M,
    :firms,
    :people,
    :A_fm,
    :At_fm,
    :cnt_m,
)

function Base.getproperty(p::GMRFProblem, name::Symbol)
    if name == :N_F
        return getfield(p, :N_firms)
    elseif name == :N_M
        return getfield(p, :N_workers)
    elseif name == :firms
        return getfield(p, :firm_ids)
    elseif name == :people
        return getfield(p, :worker_ids)
    elseif name == :A_fm
        return getfield(p, :A_prior)
    elseif name == :At_fm
        return getfield(p, :At_prior)
    elseif name == :cnt_m
        return getfield(p, :cnt_w)
    end
    return getfield(p, name)
end

"""
    HutchSLQ(; logdet_probes=30, lanczos_iters=30, cg_tol=1e-6,
             cg_maxiter=700, optim_iters=1000)

Matrix-free stochastic solver using Hutchinson stochastic Lanczos quadrature
for log determinants and preconditioned conjugate gradients for linear solves.

Use this solver when sparse direct factorization is too memory intensive.
Pass `seed` to `solve` or `gmrf_mle` for reproducible stochastic paths.
"""
struct HutchSLQ <: AbstractGMRFSolver
    logdet_probes::Int
    lanczos_iters::Int
    cg_tol::Float64
    cg_maxiter::Int
    optim_iters::Int
    function HutchSLQ(;
        logdet_probes::Int=30,
        lanczos_iters::Int=30,
        cg_tol::Float64=1e-6,
        cg_maxiter::Int=700,
        optim_iters::Int=1000,
    )
        logdet_probes > 0 || throw(ArgumentError("logdet_probes must be positive."))
        lanczos_iters > 0 || throw(ArgumentError("lanczos_iters must be positive."))
        cg_tol > 0 || throw(ArgumentError("cg_tol must be positive."))
        cg_maxiter > 0 || throw(ArgumentError("cg_maxiter must be positive."))
        optim_iters > 0 || throw(ArgumentError("optim_iters must be positive."))
        new(logdet_probes, lanczos_iters, cg_tol, cg_maxiter, optim_iters)
    end
end

"""
    ExactCholesky(; optim_iters=200, polish=true, autodiff=:finitediff)

Deterministic solver based on sparse Cholesky factorizations.

This solver is suitable for small and medium graphs where CHOLMOD fill-in is
manageable. When `polish=true`, a finite-difference L-BFGS polishing pass is
attempted after the initial Nelder-Mead optimization.
"""
struct ExactCholesky <: AbstractGMRFSolver
    optim_iters::Int
    polish::Bool
    autodiff::Symbol
    function ExactCholesky(; optim_iters::Int=200, polish::Bool=true, autodiff::Symbol=:finitediff)
        optim_iters > 0 || throw(ArgumentError("optim_iters must be positive."))
        autodiff in (:finitediff, :none) ||
            throw(ArgumentError("ExactCholesky supports autodiff=:finitediff or :none; got $(autodiff)."))
        new(optim_iters, polish, autodiff)
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
struct GMRFResult
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
    problem::GMRFProblem
    prior::AbstractGMRFPrior
    solver::AbstractGMRFSolver
    theta_unconstrained::Vector{Float64}
    metadata::NamedTuple
end

"""
Cached covariance factorization returned by `prior_covariance` or
`posterior_covariance`.

Pass a `CovarianceOperator` to `cov_block` to extract covariance blocks for
selected firm and worker IDs.
"""
struct CovarianceOperator
    kind::Symbol
    factor::Any
    result::GMRFResult
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
