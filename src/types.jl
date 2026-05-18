abstract type AbstractGMRFPrior end
abstract type AbstractGMRFSolver end

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

struct UnnormalizedPrior <: AbstractGMRFPrior
    prior_adjacency::Symbol
    function UnnormalizedPrior(; prior_adjacency::Symbol=:binary)
        prior_adjacency in (:binary, :counts) ||
            throw(ArgumentError("prior_adjacency must be :binary or :counts; got $(prior_adjacency)."))
        new(prior_adjacency)
    end
end

struct SpectralPrior <: AbstractGMRFPrior
    prior_adjacency::Symbol
    seed::Int
    function SpectralPrior(; prior_adjacency::Symbol=:binary, seed::Int=12345)
        prior_adjacency in (:binary, :counts) ||
            throw(ArgumentError("prior_adjacency must be :binary or :counts; got $(prior_adjacency)."))
        new(prior_adjacency, seed)
    end
end

struct VarianceStablePrior <: AbstractGMRFPrior end

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

struct CovarianceOperator
    kind::Symbol
    factor::Any
    result::GMRFResult
    units::Symbol
end

struct CovarianceBlock
    matrix::Matrix{Float64}
    rows::Vector{Any}
    cols::Vector{Any}
    kind::Symbol
    units::Symbol
end
