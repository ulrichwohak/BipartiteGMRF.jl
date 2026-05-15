#!/usr/bin/env julia
# ============================================================
# gmrfmle_exact.jl
#
# GMRF Maximum Likelihood Estimation using exact sparse Cholesky.
# Replaces the Hutch/SLQ + PCG approach in gmrfmle.jl with
# CHOLMOD sparse Cholesky for log-determinants and linear solves,
# and L-BFGS for optimization.
#
# Usage:
#   julia --project=. src/estimate/gmrfmle_exact.jl <path/to/edgelist.parquet> [outcome] [flags]
#
# Flags:
#   --prior_adjacency=<mode>  Prior graph adjacency: binary (default), counts
#   --obs_weighting=<mode>    Observation model: raw (default), edge, effective
#   --rho_eps=<v|estimate>    Within-match residual correlation for effective weighting
#   --decomp_target=<mode>    Variance target: estimation (default), personyear, edge
#
# Input:
#   temp/chunks/<chunk>/edgelist.parquet
#   OR temp/samples/<chunk>/<sample>/edgelist.parquet
#
# Output:
#   output/gmrfmle/<chunk>/exact/estimates.txt
#   OR output/gmrfmle/<chunk>/<sample>/estimates.txt
# ============================================================

using Parquet2
using DataFrames
using Optim
import ADTypes
using SparseArrays
using LinearAlgebra
using Statistics
using Random: MersenneTwister, rand
using Printf: @printf, @sprintf

to_float_nan(x) = x isa Missing ? NaN : Float64(x)

function build_V_stats(
    f_rows::Vector{Int},
    m_cols::Vector{Int},
    y::Vector{Float64},
    N_f::Int,
    N_m::Int
)
    K = length(y)
    @assert length(f_rows) == K
    @assert length(m_cols) == K

    proj_f = zeros(Float64, N_f)
    proj_m = zeros(Float64, N_m)
    cnt_f  = zeros(Float64, N_f)
    cnt_m  = zeros(Float64, N_m)

    @inbounds for k in 1:K
        f = f_rows[k]
        m = m_cols[k]
        val = y[k]
        cnt_f[f] += 1.0
        cnt_m[m] += 1.0
        proj_f[f] += val
        proj_m[m] += val
    end

    projected_y = vcat(proj_f, proj_m)
    A_obs = sparse(f_rows, m_cols, ones(Float64, K), N_f, N_m)
    At_obs = copy(transpose(A_obs))
    VtV = [spdiagm(0 => cnt_f)  A_obs;
           At_obs                spdiagm(0 => cnt_m)]

    return projected_y, VtV, cnt_f, cnt_m, A_obs, At_obs
end

function build_weighted_V_stats(
    f_rows::Vector{Int},
    m_cols::Vector{Int},
    y::Vector{Float64},
    w::Vector{Float64},
    N_f::Int,
    N_m::Int
)
    K = length(y)
    @assert length(f_rows) == K
    @assert length(m_cols) == K
    @assert length(w) == K

    proj_f = zeros(Float64, N_f)
    proj_m = zeros(Float64, N_m)
    cnt_f  = zeros(Float64, N_f)
    cnt_m  = zeros(Float64, N_m)
    ydot   = 0.0

    @inbounds for k in 1:K
        f = f_rows[k]
        m = m_cols[k]
        wk = w[k]
        val = y[k]
        cnt_f[f] += wk
        cnt_m[m] += wk
        proj_f[f] += wk * val
        proj_m[m] += wk * val
        ydot += wk * val * val
    end

    projected_y = vcat(proj_f, proj_m)
    A_obs = sparse(f_rows, m_cols, w, N_f, N_m)
    At_obs = copy(transpose(A_obs))
    VtV = [spdiagm(0 => cnt_f)  A_obs;
           At_obs                spdiagm(0 => cnt_m)]

    return projected_y, VtV, cnt_f, cnt_m, A_obs, At_obs, ydot
end

rhoeps_from_unconstrained(u::Real) = 0.999 / (1.0 + exp(-Float64(u)))
rhoeps_to_unconstrained(r::Real) = begin
    x = Float64(r) / 0.999
    x = min(max(x, 1e-12), 1.0 - 1e-12)
    log(x / (1.0 - x))
end

function effective_match_weights(T::AbstractVector{<:Real}, rho_eps::Float64)
    0.0 <= rho_eps < 1.0 || error("rho_eps must satisfy 0 <= rho_eps < 1, got $rho_eps")
    return Float64.(T) ./ (1.0 .+ rho_eps .* (Float64.(T) .- 1.0))
end

function edge_ssw(x)
    μ = mean(x)
    s = 0.0
    @inbounds for v in x
        d = Float64(v) - μ
        s += d * d
    end
    return s
end

function safe_float_id(x::Real)
    s = @sprintf("%.6g", Float64(x))
    return replace(replace(s, "-" => "m"), "." => "p")
end

function build_match_weight_stats(
    f_rows::Vector{Int},
    m_cols::Vector{Int},
    y::Vector{Float64},
    T::Vector{Int},
    N_f::Int,
    N_m::Int,
    rho_eps::Float64
)
    w = effective_match_weights(T, rho_eps)
    projected_y, VtV, cnt_f, cnt_m, A_obs, At_obs, ydot =
        build_weighted_V_stats(f_rows, m_cols, y, w, N_f, N_m)
    return (projected_y=projected_y, VtV=VtV, cnt_f=cnt_f, cnt_m=cnt_m,
            A_obs=A_obs, At_obs=At_obs, ydot=ydot,
            log_weight_sum=sum(log, w), effective_weight_sum=sum(w),
            mean_effective_weight=mean(w), max_effective_weight=maximum(w),
            effective_weight_over_T_sum=sum(w ./ Float64.(T)))
end

function normalize_decomp_target(target::Symbol)
    target in (:estimation, :likelihood, :likelihood_mean) && return :estimation
    target in (:personyear, :edge) && return target
    error("Unknown decomp_target: $(target). Use estimation, personyear, or edge.")
end

function target_weight_vector(prep, target::Symbol)
    t = normalize_decomp_target(target)
    T = Float64.(prep.decomp_T)
    if t == :estimation
        if prep.obs_weighting == :raw
            return T, :annual
        elseif prep.obs_weighting == :edge
            return ones(Float64, length(T)), :mean
        else
            return effective_match_weights(prep.decomp_T, prep.rho_eps_likelihood), :mean
        end
    elseif t == :personyear
        return T, :annual
    else
        return ones(Float64, length(T)), :mean
    end
end

function decomp_target_stats(prep, target::Symbol)
    t = normalize_decomp_target(target)
    w, residual_level = target_weight_vector(prep, t)
    projected_y, VtV, cnt_f, cnt_m, A_obs, At_obs, ydot_mean =
        build_weighted_V_stats(
            prep.decomp_f_rows, prep.decomp_m_cols, prep.decomp_y, w,
            prep.N_F, prep.N_M
        )

    W = sum(w)
    T = Float64.(prep.decomp_T)
    ydot_total = residual_level == :annual ?
        ydot_mean + prep.personyear_within_ss :
        ydot_mean
    observed_second_moment = prep.y_std^2 * ydot_total / W

    return (target=t, residual_level=residual_level,
            weights=w, weight_sum=W, weight_over_T_sum=sum(w ./ T),
            projected_y=projected_y, VtV=VtV,
            cnt_f=cnt_f, cnt_m=cnt_m, A_obs=A_obs, At_obs=At_obs,
            ydot_mean=ydot_mean, ydot_total=ydot_total,
            observed_second_moment=observed_second_moment)
end

function residual_decomp_components(prep, σe_original::Float64, target_stats)
    σeps2 = σe_original^2
    mean_factor = target_stats.weight_over_T_sum / target_stats.weight_sum

    if prep.rho_eps_likelihood !== nothing
        ρe = prep.rho_eps_likelihood
        V_eta = ρe * σeps2
        V_u_annual = (1.0 - ρe) * σeps2
        V_u_target = target_stats.residual_level == :annual ?
            V_u_annual :
            V_u_annual * mean_factor
        V_eps_target = V_eta + V_u_target
        return (rho_eps=ρe, V_eta_match=V_eta, V_u_annual=V_u_annual,
                V_u_target=V_u_target, V_eps_target=V_eps_target,
                mean_residual_factor=mean_factor,
                model=:compound_symmetric)
    end

    if prep.obs_weighting == :raw && target_stats.residual_level == :mean
        V_u_target = σeps2 * mean_factor
    else
        V_u_target = σeps2
    end
    return (rho_eps=nothing, V_eta_match=0.0, V_u_annual=σeps2,
            V_u_target=V_u_target, V_eps_target=V_u_target,
            mean_residual_factor=mean_factor,
            model=:iid)
end

# ============================================================
# 1) Prepare data
# ============================================================

function prepare_data(
    df::DataFrame;
    outcome::Symbol,
    prior_adjacency::Symbol=:binary,
    obs_weighting::Symbol=:raw,
    rho_eps_fixed::Union{Float64,Nothing}=nothing,
    rho_eps_estimate::Bool=false,
    verbose::Bool=true
)
    required = [:frame_id_numeric, :person_id, outcome]
    for c in required
        hasproperty(df, c) || error("Missing required column: $(c)")
    end

    nrow(df) == 0 && error("Empty dataset.")

    if verbose
        @printf("Raw rows: %d\n", nrow(df))
        nunique = length(unique(zip(df.frame_id_numeric, df.person_id)))
        if nunique != nrow(df)
            @printf("WARNING: duplicate (firm,person) pairs: rows=%d, unique_edges=%d\n",
                    nrow(df), nunique)
        else
            @printf("Unique edges: %d\n", nunique)
        end
    end

    obs_weighting in (:raw, :edge, :effective) ||
        error("Unknown obs_weighting: $(obs_weighting). Use raw, edge, or effective.")
    if obs_weighting != :effective && (rho_eps_fixed !== nothing || rho_eps_estimate)
        error("rho_eps is only meaningful with obs_weighting=:effective.")
    end
    if obs_weighting == :effective && !rho_eps_estimate && rho_eps_fixed === nothing
        error("obs_weighting=:effective requires rho_eps_fixed or rho_eps_estimate=true.")
    end
    if rho_eps_fixed !== nothing && !(0.0 <= rho_eps_fixed < 1.0)
        error("rho_eps must satisfy 0 <= rho_eps < 1, got $rho_eps_fixed")
    end

    y_raw_all = map(to_float_nan, df[!, outcome])
    keep = map(isfinite, y_raw_all)
    d = df[keep, :]
    y_raw = Float64.(y_raw_all[keep])
    personyear_rows = length(y_raw)

    if verbose
        @printf("After non-missing %s filter: K=%d\n", String(outcome), length(y_raw))
    end

    length(y_raw) == 0 && error("No usable observations after filtering non-missing $(outcome).")

    μy = mean(y_raw)
    y_std = std(y_raw)
    (isfinite(y_std) && y_std > 0) || error("Outcome has zero/invalid std. Cannot scale.")

    d_work_all = DataFrame(
        frame_id_numeric=d.frame_id_numeric,
        person_id=d.person_id,
        __y=y_raw
    )
    collapsed_all = combine(groupby(d_work_all, [:frame_id_numeric, :person_id]),
        :__y => length => :T,
        :__y => mean => :y_mean,
        :__y => edge_ssw => :ssw_raw
    )

    local firms, people, N_F, N_M, K
    local f_rows, m_cols, y, ydot
    local projected_y, VtV, cnt_f, cnt_m, A_obs, At_obs
    local unique_edges, duplicate_rows, mean_edge_count, max_edge_count
    local T_edge, within_ss, within_df, log_weight_sum, effective_weight_sum
    local effective_weight_over_T_sum
    local mean_effective_weight, max_effective_weight, rho_eps_likelihood
    local decomp_f_rows, decomp_m_cols, decomp_y, decomp_T, personyear_within_ss
    local A_prior_base

    if obs_weighting == :raw
        firms  = unique(d.frame_id_numeric)
        people = unique(d.person_id)

        N_F = length(firms)
        N_M = length(people)
        K   = nrow(d)

        if verbose
            @printf("Indexing IDs... N_F=%d, N_M=%d, K=%d\n", N_F, N_M, K)
        end

        firm_to_idx   = Dict{eltype(firms),Int}(f => i for (i,f) in enumerate(firms))
        person_to_idx = Dict{eltype(people),Int}(p => i for (i,p) in enumerate(people))

        f_rows = Vector{Int}(undef, K)
        m_cols = Vector{Int}(undef, K)
        @inbounds for k in 1:K
            f_rows[k] = firm_to_idx[d.frame_id_numeric[k]]
            m_cols[k] = person_to_idx[d.person_id[k]]
        end

        y = (y_raw .- μy) ./ y_std
        ydot = dot(y, y)

        projected_y, VtV, cnt_f, cnt_m, A_obs, At_obs =
            build_V_stats(f_rows, m_cols, y, N_F, N_M)
        A_prior_base = copy(A_obs)

        decomp_f_rows = Vector{Int}(undef, nrow(collapsed_all))
        decomp_m_cols = Vector{Int}(undef, nrow(collapsed_all))
        @inbounds for k in 1:nrow(collapsed_all)
            decomp_f_rows[k] = firm_to_idx[collapsed_all.frame_id_numeric[k]]
            decomp_m_cols[k] = person_to_idx[collapsed_all.person_id[k]]
        end
        decomp_y = (Float64.(collapsed_all.y_mean) .- μy) ./ y_std
        decomp_T = Int.(collapsed_all.T)
        personyear_within_ss = sum(Float64.(collapsed_all.ssw_raw)) / (y_std^2)

        unique_edges = nnz(A_obs)
        duplicate_rows = K - unique_edges
        mean_edge_count = unique_edges > 0 ? Float64(K) / Float64(unique_edges) : NaN
        max_edge_count = unique_edges > 0 ? maximum(A_obs.nzval) : NaN
        T_edge = ones(Int, K)
        within_ss = 0.0
        within_df = 0
        log_weight_sum = 0.0
        effective_weight_sum = Float64(K)
        effective_weight_over_T_sum = Float64(K)
        mean_effective_weight = 1.0
        max_effective_weight = 1.0
        rho_eps_likelihood = nothing
    else
        collapsed = collapsed_all

        firms  = unique(collapsed.frame_id_numeric)
        people = unique(collapsed.person_id)

        N_F = length(firms)
        N_M = length(people)
        K   = nrow(collapsed)

        if verbose
            @printf("Collapsed to match means: N_F=%d, N_M=%d, K_edges=%d, finite person-years=%d\n",
                    N_F, N_M, K, personyear_rows)
        end

        firm_to_idx   = Dict{eltype(firms),Int}(f => i for (i,f) in enumerate(firms))
        person_to_idx = Dict{eltype(people),Int}(p => i for (i,p) in enumerate(people))

        f_rows = Vector{Int}(undef, K)
        m_cols = Vector{Int}(undef, K)
        @inbounds for k in 1:K
            f_rows[k] = firm_to_idx[collapsed.frame_id_numeric[k]]
            m_cols[k] = person_to_idx[collapsed.person_id[k]]
        end

        y = (Float64.(collapsed.y_mean) .- μy) ./ y_std
        T_edge = Int.(collapsed.T)
        decomp_f_rows = copy(f_rows)
        decomp_m_cols = copy(m_cols)
        decomp_y = copy(y)
        decomp_T = copy(T_edge)
        personyear_within_ss = sum(Float64.(collapsed.ssw_raw)) / (y_std^2)
        A_prior_base = sparse(f_rows, m_cols, Float64.(T_edge), N_F, N_M)

        unique_edges = K
        duplicate_rows = personyear_rows - K
        mean_edge_count = mean(Float64.(T_edge))
        max_edge_count = maximum(Float64.(T_edge))

        if obs_weighting == :edge
            w = ones(Float64, K)
            projected_y, VtV, cnt_f, cnt_m, A_obs, At_obs, ydot =
                build_weighted_V_stats(f_rows, m_cols, y, w, N_F, N_M)
            within_ss = 0.0
            within_df = 0
            log_weight_sum = 0.0
            effective_weight_sum = Float64(K)
            effective_weight_over_T_sum = Float64(K)
            mean_effective_weight = 1.0
            max_effective_weight = 1.0
            rho_eps_likelihood = nothing
        else
            rho0 = rho_eps_estimate ? 0.5 : rho_eps_fixed::Float64
            stats = build_match_weight_stats(f_rows, m_cols, y, T_edge, N_F, N_M, rho0)
            projected_y = stats.projected_y
            VtV = stats.VtV
            cnt_f = stats.cnt_f
            cnt_m = stats.cnt_m
            A_obs = stats.A_obs
            At_obs = stats.At_obs
            ydot = stats.ydot
            log_weight_sum = stats.log_weight_sum
            effective_weight_sum = stats.effective_weight_sum
            effective_weight_over_T_sum = stats.effective_weight_over_T_sum
            mean_effective_weight = stats.mean_effective_weight
            max_effective_weight = stats.max_effective_weight
            within_ss = personyear_within_ss
            within_df = personyear_rows - K
            rho_eps_likelihood = rho0
        end
    end

    if verbose
        @printf("Outcome scaling: mean removed, y_std=%.6g\n", y_std)
        @printf("Observation weighting: %s", String(obs_weighting))
        if obs_weighting == :effective
            mode = rho_eps_estimate ? "estimated" : @sprintf("fixed %.6f", rho_eps_fixed)
            @printf(" | rho_eps=%s | within_df=%d | effective_weight_sum=%.3f",
                    mode, within_df, effective_weight_sum)
        end
        @printf("\n")
    end

    prior_adjacency in (:binary, :counts) ||
        error("Unknown prior_adjacency: $(prior_adjacency). Use :binary or :counts.")

    # Bipartite adjacency for the GMRF prior. A_obs is the observation-weighted
    # measurement matrix; A_fm is the prior graph and can be binary or counts.
    A_fm = copy(A_prior_base)
    if prior_adjacency == :binary
        A_fm.nzval .= 1.0
    end

    d_f = vec(sum(A_fm; dims=2))
    d_m = vec(sum(A_fm; dims=1))
    any(d_f .<= 0) && error("Zero-degree firm detected.")
    any(d_m .<= 0) && error("Zero-degree person detected.")

    df_is = 1.0 ./ sqrt.(d_f)
    dm_is = 1.0 ./ sqrt.(d_m)

    # Degree-normalized weight matrix W = D_f^{-1/2} A D_m^{-1/2}
    W = spdiagm(0 => df_is) * A_fm * spdiagm(0 => dm_is)
    Wt = copy(transpose(W))
    At_fm = copy(transpose(A_fm))

    if verbose
        @printf("Building sparse structures... nnz(W)=%d\n", nnz(W))
        @printf("Prior adjacency: %s | unique_edges=%d duplicate_rows=%d mean_edge_count=%.3f max_edge_count=%.0f\n",
                String(prior_adjacency), unique_edges, duplicate_rows, mean_edge_count, max_edge_count)
        @printf("Prior graph degrees: max_firm=%.0f max_manager=%.0f total_prior_weight=%.0f\n",
                maximum(d_f), maximum(d_m), sum(A_fm.nzval))
    end

    return (y=y, ydot=ydot, y_std=y_std,
            W=W, Wt=Wt, A_fm=A_fm, At_fm=At_fm,
            d_f=d_f, d_m=d_m,
            prior_adjacency=prior_adjacency,
            obs_weighting=obs_weighting,
            rho_eps_fixed=rho_eps_fixed,
            rho_eps_estimate=rho_eps_estimate,
            rho_eps_likelihood=rho_eps_likelihood,
            personyear_rows=personyear_rows,
            within_ss=within_ss,
            within_df=within_df,
            log_weight_sum=log_weight_sum,
            effective_weight_sum=effective_weight_sum,
            effective_weight_over_T_sum=effective_weight_over_T_sum,
            mean_effective_weight=mean_effective_weight,
            max_effective_weight=max_effective_weight,
            unique_edges=unique_edges, duplicate_rows=duplicate_rows,
            mean_edge_count=mean_edge_count, max_edge_count=max_edge_count,
            total_prior_weight=sum(A_fm.nzval),
            max_prior_degree_f=maximum(d_f), max_prior_degree_m=maximum(d_m),
            cnt_f=cnt_f, cnt_m=cnt_m,
            A_obs=A_obs, At_obs=At_obs,
            projected_y=projected_y, VtV=VtV,
            base_f_rows=f_rows, base_m_cols=m_cols, base_y=y, base_T=T_edge,
            decomp_f_rows=decomp_f_rows, decomp_m_cols=decomp_m_cols,
            decomp_y=decomp_y, decomp_T=decomp_T,
            personyear_within_ss=personyear_within_ss,
            K=K, N_F=N_F, N_M=N_M)
end

# ============================================================
# 2) Exact NLL via sparse Cholesky (fresh construction each call)
# ============================================================

"""
    nll_exact(params, ydot, projected_y, K, W, Wt, VtV, N_F, N_M) -> Float64

Negative log-likelihood using exact sparse Cholesky for log-determinants
and direct solves for the quadratic form.

NLL = 0.5 * [K * 2log(σ_ε) + (logdet M - logdet Q) + λ y'y - λ² (V'y)' M⁻¹ (V'y)]

where M = Q + λ V'V.
"""
function nll_exact(
    params::Vector{Float64},
    ydot::Float64,
    projected_y::Vector{Float64},
    K::Int,
    W::SparseMatrixCSC{Float64,Int},
    Wt::SparseMatrixCSC{Float64,Int},
    VtV::SparseMatrixCSC{Float64,Int},
    N_F::Int, N_M::Int;
    a_weighting::Symbol=:degree,
    d_f::Vector{Float64}=Float64[],
    d_m::Vector{Float64}=Float64[],
    A_fm::SparseMatrixCSC{Float64,Int}=spzeros(0,0),
    At_fm::SparseMatrixCSC{Float64,Int}=spzeros(0,0),
    log_weight_sum::Real=0.0,
    within_ss::Real=0.0,
    within_df::Int=0,
    rho_eps::Union{Nothing,Float64}=nothing
)
    BIG = 1e30

    ρ  = 0.99 * tanh(params[1])
    σa = exp(params[2])
    σz = exp(params[3])
    σe = exp(params[4])

    if !(isfinite(ρ) && isfinite(σa) && isfinite(σz) && isfinite(σe)) ||
       σe <= 0 || σa <= 0 || σz <= 0
        return BIG
    end

    λ       = 1.0 / (σe^2)
    inv_sa2 = 1.0 / (σa^2)
    inv_sz2 = 1.0 / (σz^2)
    cross   = ρ / (σa * σz)

    # Build Q
    #   degree:     Q = [inv_sa2 * I,        -cross * W;   -cross * W',        inv_sz2 * I]
    #   unweighted: Q = [inv_sa2 * D_f,      -cross * A;   -cross * A',        inv_sz2 * D_m]
    if a_weighting == :degree
        Q = [spdiagm(0 => fill(inv_sa2, N_F))  (-cross .* W);
             (-cross .* Wt)                      spdiagm(0 => fill(inv_sz2, N_M))]
    else  # :unweighted
        Q = [spdiagm(0 => inv_sa2 .* d_f)  (-cross .* A_fm);
             (-cross .* At_fm)               spdiagm(0 => inv_sz2 .* d_m)]
    end

    # Build M = Q + λ V'V
    M = Q + λ .* VtV

    # Sparse Cholesky factorizations
    local FQ, FM
    try
        FQ = cholesky(Symmetric(Q))
    catch
        return BIG
    end

    try
        FM = cholesky(Symmetric(M))
    catch
        return BIG
    end

    ldQ = logdet(FQ)
    ldM = logdet(FM)

    if !(isfinite(ldQ) && isfinite(ldM))
        return BIG
    end

    # Solve M x = V'y
    xsol = FM \ projected_y

    quad_term = dot(projected_y, xsol)
    if !isfinite(quad_term)
        return BIG
    end

    residual_corr_term = 0.0
    if rho_eps !== nothing
        if !(0.0 <= rho_eps < 1.0)
            return BIG
        end
        omr = 1.0 - rho_eps
        residual_corr_term =
            Float64(within_df) * (2.0 * log(σe) + log(omr)) +
            λ * Float64(within_ss) / omr
    end

    val = 0.5 * (
        K * 2.0 * log(σe) - Float64(log_weight_sum) +
        (ldM - ldQ) + λ * ydot - (λ^2) * quad_term +
        residual_corr_term
    )
    return isfinite(val) ? val : BIG
end

# ============================================================
# 3) Exact variance decomposition via Hutchinson + Cholesky
# ============================================================

"""
    exact_variance_decomp(prep, p̂; m, seed, verbose)

    Prior-implied variance decomposition using Hutchinson probes with exact
    Q⁻¹ solves:

    (1/K) tr(Var(y)) = V_firm + V_manager + V_cross + V_eps

where

    V_firm    = (y_std² / K) · tr(diag(cnt_f) · Q⁻¹)
    V_manager = (y_std² / K) · tr(diag(cnt_m) · Q⁻¹)
    V_cross   = (y_std² / K) · tr([0 A_obs; A_obs' 0] · Q⁻¹)   [signed]
    V_eps     = sigma_eps²

This is the AKM-style count-weighted decomposition (Object 2 in
`gmrfmle.jl`), so repeated person-year observations receive the same weights
used by AKM/KSS.

Each Rademacher probe v → u = FQ \\ v (exact triangular solve from CHOLMOD
factorization).  Always converges; no PCG failures to guard against.
"""
function exact_variance_decomp(
    prep,
    p̂::Vector{Float64};
    target::Symbol=:estimation,
    m::Int=50,
    seed::Int=20260126,
    verbose::Bool=true,
    a_weighting::Symbol=:degree
)
    ρ  = 0.99 * tanh(p̂[1])
    σa = exp(p̂[2])
    σz = exp(p̂[3])
    σe = exp(p̂[4])

    inv_sa2 = 1.0 / σa^2
    inv_sz2 = 1.0 / σz^2
    cross   = ρ / (σa * σz)

    n = prep.N_F + prep.N_M

    # Build Q at MLE estimates (same formula as nll_exact)
    if a_weighting == :degree
        Q = [spdiagm(0 => fill(inv_sa2, prep.N_F))  (-cross .* prep.W);
             (-cross .* prep.Wt)                      spdiagm(0 => fill(inv_sz2, prep.N_M))]
    else  # :unweighted
        Q = [spdiagm(0 => inv_sa2 .* prep.d_f)  (-cross .* prep.A_fm);
             (-cross .* prep.At_fm)               spdiagm(0 => inv_sz2 .* prep.d_m)]
    end

    local FQ
    try
        FQ = cholesky(Symmetric(Q))
    catch
        @warn "exact_variance_decomp: Q Cholesky failed at MLE estimates"
        return nothing
    end

    rng = MersenneTwister(seed + 77_777)
    dstats = decomp_target_stats(prep, target)
    cnt_f = dstats.cnt_f
    cnt_m = dstats.cnt_m

    acc_F     = 0.0
    acc_M     = 0.0
    acc_cross = 0.0

    v    = Vector{Float64}(undef, n)
    wv_f = Vector{Float64}(undef, prep.N_F)
    wv_m = Vector{Float64}(undef, prep.N_M)

    verbose && @printf("\n--- Variance decomposition (target=%s, exact Cholesky, m=%d) ---\n",
                       String(dstats.target), m)

    t_decomp = @elapsed for _ in 1:m
        @inbounds for i in 1:n
            v[i] = rand(rng, Bool) ? 1.0 : -1.0
        end

        # Exact solve: u = FQ \ v  (no convergence failure possible)
        u = FQ \ v

        vf = view(v, 1:prep.N_F)
        vm = view(v, prep.N_F+1:n)
        uf = view(u, 1:prep.N_F)
        um = view(u, prep.N_F+1:n)

        @inbounds for i in 1:prep.N_F
            acc_F += cnt_f[i] * vf[i] * uf[i]
        end
        @inbounds for j in 1:prep.N_M
            acc_M += cnt_m[j] * vm[j] * um[j]
        end

        mul!(wv_f, dstats.A_obs,  um)
        mul!(wv_m, dstats.At_obs, uf)
        acc_cross += dot(vf, wv_f) + dot(vm, wv_m)
    end

    scale     = prep.y_std^2 / (Float64(m) * dstats.weight_sum)
    V_firm    = acc_F     * scale
    V_manager = acc_M     * scale
    V_cross   = acc_cross * scale
    resid     = residual_decomp_components(prep, σe * prep.y_std, dstats)
    V_eps     = resid.V_eps_target
    V_total   = V_firm + V_manager + V_cross + V_eps

    if verbose
        @printf("Decomp time: %.1fs\n", t_decomp)
        @printf("  Target weight sum = %.6f | observed second moment = %.6f\n",
                dstats.weight_sum, dstats.observed_second_moment)
        @printf("  V_firm    = %+.6f\n", V_firm)
        @printf("  V_manager = %+.6f\n", V_manager)
        @printf("  V_cross   = %+.6f  (= 2 x sorting cov; positive when rho > 0)\n", V_cross)
        if resid.model == :compound_symmetric
            @printf("  V_eta     = %+.6f  (persistent match residual)\n", resid.V_eta_match)
            @printf("  V_u       = %+.6f  (target annual/idiosyncratic residual contribution)\n",
                    resid.V_u_target)
        end
        @printf("  V_eps     = %+.6f  (target residual contribution)\n", V_eps)
        @printf("  V_total   = %+.6f  (target observed second moment = %.6f)\n",
                V_total, dstats.observed_second_moment)
    end

    return (V_firm=V_firm, V_manager=V_manager, V_cross=V_cross,
            V_eps=V_eps, V_eps_target=V_eps, V_total=V_total,
            V_eta_match=resid.V_eta_match,
            V_u_annual=resid.V_u_annual,
            V_u_target=resid.V_u_target,
            residual_model=resid.model,
            mean_residual_factor=resid.mean_residual_factor,
            target=dstats.target,
            residual_level=dstats.residual_level,
            weight_sum=dstats.weight_sum,
            observed_second_moment=dstats.observed_second_moment,
            probes=m)
end

"""
    exact_posterior_decomp(prep, p̂; m, seed, verbose, a_weighting)

Posterior expected AKM variance decomposition using exact CHOLMOD solves for
M = Q + λV'V and Hutchinson traces. This is directly comparable to AKM/KSS on
the same estimation sample:

    E[sigma_a^2 | y], E[sigma_z^2 | y], E[sigma_az | y], sigma_eps^2

with AKM-style count weights.
"""
function exact_posterior_decomp(
    prep,
    p̂::Vector{Float64};
    target::Symbol=:estimation,
    m::Int=50,
    seed::Int=20260126,
    verbose::Bool=true,
    a_weighting::Symbol=:degree
)
    ρ  = 0.99 * tanh(p̂[1])
    σa = exp(p̂[2])
    σz = exp(p̂[3])
    σe = exp(p̂[4])

    λ       = 1.0 / σe^2
    inv_sa2 = 1.0 / σa^2
    inv_sz2 = 1.0 / σz^2
    cross   = ρ / (σa * σz)

    n = prep.N_F + prep.N_M

    if a_weighting == :degree
        Q = [spdiagm(0 => fill(inv_sa2, prep.N_F))  (-cross .* prep.W);
             (-cross .* prep.Wt)                      spdiagm(0 => fill(inv_sz2, prep.N_M))]
    else  # :unweighted
        Q = [spdiagm(0 => inv_sa2 .* prep.d_f)  (-cross .* prep.A_fm);
             (-cross .* prep.At_fm)               spdiagm(0 => inv_sz2 .* prep.d_m)]
    end
    M = Q + λ .* prep.VtV

    local FM
    try
        FM = cholesky(Symmetric(M))
    catch
        @warn "exact_posterior_decomp: M Cholesky failed at MLE estimates"
        return nothing
    end

    rng = MersenneTwister(seed + 88_888)
    dstats = decomp_target_stats(prep, target)
    cnt_f = dstats.cnt_f
    cnt_m = dstats.cnt_m

    rhs       = λ .* prep.projected_y
    theta_hat = FM \ rhs
    th_f      = view(theta_hat, 1:prep.N_F)
    th_m      = view(theta_hat, prep.N_F+1:n)

    tmp_Aobs_thm = Vector{Float64}(undef, prep.N_F)
    mul!(tmp_Aobs_thm, dstats.A_obs, th_m)

    qa  = dot(cnt_f, th_f .^ 2)
    qz  = dot(cnt_m, th_m .^ 2)
    qaz = dot(th_f, tmp_Aobs_thm)

    acc_F     = 0.0
    acc_M     = 0.0
    acc_cross = 0.0

    v    = Vector{Float64}(undef, n)
    wv_f = Vector{Float64}(undef, prep.N_F)
    wv_m = Vector{Float64}(undef, prep.N_M)

    verbose && @printf("\n--- Posterior decomposition (target=%s, exact Cholesky, m=%d) ---\n",
                       String(dstats.target), m)

    t_decomp = @elapsed for _ in 1:m
        @inbounds for i in 1:n
            v[i] = rand(rng, Bool) ? 1.0 : -1.0
        end

        u = FM \ v

        vf = view(v, 1:prep.N_F)
        vm = view(v, prep.N_F+1:n)
        uf = view(u, 1:prep.N_F)
        um = view(u, prep.N_F+1:n)

        @inbounds for i in 1:prep.N_F
            acc_F += cnt_f[i] * vf[i] * uf[i]
        end
        @inbounds for j in 1:prep.N_M
            acc_M += cnt_m[j] * vm[j] * um[j]
        end
        mul!(wv_f, dstats.A_obs,  um)
        mul!(wv_m, dstats.At_obs, uf)
        acc_cross += dot(vf, wv_f) + dot(vm, wv_m)
    end

    quad_scale  = prep.y_std^2 / dstats.weight_sum
    trace_scale = prep.y_std^2 / (Float64(m) * dstats.weight_sum)

    E_sa2_quad  = qa  * quad_scale
    E_sa2_trace = acc_F     * trace_scale
    E_sa2       = E_sa2_quad + E_sa2_trace

    E_sz2_quad  = qz  * quad_scale
    E_sz2_trace = acc_M     * trace_scale
    E_sz2       = E_sz2_quad + E_sz2_trace

    E_saz_quad  = qaz * quad_scale
    E_saz_trace = acc_cross * trace_scale / 2.0
    E_saz       = E_saz_quad + E_saz_trace

    resid = residual_decomp_components(prep, σe * prep.y_std, dstats)
    V_eps = resid.V_eps_target
    E_tot = E_sa2 + E_sz2 + 2.0 * E_saz + V_eps

    if verbose
        @printf("Decomp time: %.1fs\n", t_decomp)
        @printf("  Target weight sum = %.6f | observed second moment = %.6f\n",
                dstats.weight_sum, dstats.observed_second_moment)
        @printf("  E[sigma_a^2|y] = %+.6f  (quad=%+.6f, trace=%+.6f)\n",
                E_sa2, E_sa2_quad, E_sa2_trace)
        @printf("  E[sigma_z^2|y] = %+.6f  (quad=%+.6f, trace=%+.6f)\n",
                E_sz2, E_sz2_quad, E_sz2_trace)
        @printf("  E[sigma_az |y] = %+.6f  (quad=%+.6f, trace=%+.6f)\n",
                E_saz, E_saz_quad, E_saz_trace)
        if resid.model == :compound_symmetric
            @printf("  V_eta          = %+.6f  (persistent match residual)\n", resid.V_eta_match)
            @printf("  V_u            = %+.6f  (target annual/idiosyncratic residual contribution)\n",
                    resid.V_u_target)
        end
        @printf("  V_eps          = %+.6f  (target residual contribution)\n", V_eps)
        @printf("  E_total        = %+.6f  (target observed second moment = %.6f)\n",
                E_tot, dstats.observed_second_moment)
    end

    return (E_sa2=E_sa2, E_sz2=E_sz2, E_saz=E_saz,
            E_sa2_quad=E_sa2_quad, E_sa2_trace=E_sa2_trace,
            E_sz2_quad=E_sz2_quad, E_sz2_trace=E_sz2_trace,
            E_saz_quad=E_saz_quad, E_saz_trace=E_saz_trace,
            V_eps=V_eps, V_eps_target=V_eps, E_total=E_tot,
            V_eta_match=resid.V_eta_match,
            V_u_annual=resid.V_u_annual,
            V_u_target=resid.V_u_target,
            residual_model=resid.model,
            mean_residual_factor=resid.mean_residual_factor,
            target=dstats.target,
            residual_level=dstats.residual_level,
            weight_sum=dstats.weight_sum,
            observed_second_moment=dstats.observed_second_moment,
            probes=m)
end

function akm_comparable_summary(post)
    V_firm    = post.E_sa2
    V_manager = post.E_sz2
    V_cross   = 2.0 * post.E_saz
    V_eps     = post.V_eps
    V_total   = post.E_total
    sigma_a   = V_firm > 0 ? sqrt(V_firm) : NaN
    sigma_z   = V_manager > 0 ? sqrt(V_manager) : NaN
    sigma_eps = V_eps >= 0 ? sqrt(V_eps) : NaN
    rho = (isfinite(sigma_a) && isfinite(sigma_z) && sigma_a > 0 && sigma_z > 0) ?
          post.E_saz / (sigma_a * sigma_z) : NaN
    return (rho=rho, sigma_a=sigma_a, sigma_z=sigma_z, sigma_eps=sigma_eps,
            V_firm=V_firm, V_manager=V_manager, V_cross=V_cross,
            V_eps=V_eps, V_total=V_total)
end

# ============================================================
# 4) Estimation
# ============================================================

function estimate_exact(
    df::DataFrame;
    outcome::Symbol=:lnR,
    iters::Int=1000,
    verbose::Bool=true,
    decomp_probes::Int=0,
    a_weighting::Symbol=:degree,
    prior_adjacency::Symbol=:binary,
    obs_weighting::Symbol=:raw,
    rho_eps_fixed::Union{Float64,Nothing}=nothing,
    rho_eps_estimate::Bool=false,
    decomp_target::Symbol=:estimation,
    seed::Int=20260126
)
    decomp_target = normalize_decomp_target(decomp_target)
    @printf("\n--- Preparing data ---\n")
    prep = prepare_data(df; outcome=outcome, prior_adjacency=prior_adjacency,
                        obs_weighting=obs_weighting,
                        rho_eps_fixed=rho_eps_fixed,
                        rho_eps_estimate=rho_eps_estimate,
                        verbose=verbose)

    n = prep.N_F + prep.N_M
    @printf("\n--- Model dimensions ---\n")
    @printf("N_F=%d, N_M=%d, K=%d, n=%d\n", prep.N_F, prep.N_M, prep.K, n)
    @printf("Method: exact sparse Cholesky + L-BFGS | a_weighting=%s | prior_adjacency=%s | obs_weighting=%s\n",
            String(a_weighting), String(prior_adjacency), String(obs_weighting))

    eval_counter = Ref(0)

    function stats_for_objective(p_full)
        if prep.rho_eps_estimate
            rho_eps_cur = rhoeps_from_unconstrained(p_full[5])
            stats = build_match_weight_stats(
                prep.base_f_rows, prep.base_m_cols, prep.base_y, prep.base_T,
                prep.N_F, prep.N_M, rho_eps_cur
            )
            return (ydot=stats.ydot, projected_y=stats.projected_y,
                    log_weight_sum=stats.log_weight_sum,
                    rho_eps=rho_eps_cur,
                    effective_weight_sum=stats.effective_weight_sum,
                    mean_effective_weight=stats.mean_effective_weight,
                    max_effective_weight=stats.max_effective_weight,
                    VtV=stats.VtV, cnt_f=stats.cnt_f, cnt_m=stats.cnt_m,
                    A_obs=stats.A_obs, At_obs=stats.At_obs)
        else
            return (ydot=prep.ydot, projected_y=prep.projected_y,
                    log_weight_sum=prep.log_weight_sum,
                    rho_eps=prep.rho_eps_likelihood,
                    effective_weight_sum=prep.effective_weight_sum,
                    mean_effective_weight=prep.mean_effective_weight,
                    max_effective_weight=prep.max_effective_weight,
                    VtV=prep.VtV, cnt_f=prep.cnt_f, cnt_m=prep.cnt_m,
                    A_obs=prep.A_obs, At_obs=prep.At_obs)
        end
    end

    function obj(p)
        eval_counter[] += 1
        st = stats_for_objective(p)
        if verbose && (eval_counter[] == 1 || eval_counter[] % 20 == 0)
            ρ_cur  = 0.99 * tanh(p[1])
            σa_cur = exp(p[2]) * prep.y_std
            σz_cur = exp(p[3]) * prep.y_std
            σe_cur = exp(p[4]) * prep.y_std
            if length(p) == 5
                @printf("[obj] eval=%d | ρ=%.4f σa=%.4f σz=%.4f σε=%.4f rho_eps=%.4f\n",
                        eval_counter[], ρ_cur, σa_cur, σz_cur, σe_cur, st.rho_eps)
            else
                @printf("[obj] eval=%d | ρ=%.4f σa=%.4f σz=%.4f σε=%.4f\n",
                        eval_counter[], ρ_cur, σa_cur, σz_cur, σe_cur)
            end
        end
        return nll_exact(p, st.ydot, st.projected_y, prep.K,
                         prep.W, prep.Wt, st.VtV, prep.N_F, prep.N_M;
                         a_weighting=a_weighting,
                         d_f=prep.d_f, d_m=prep.d_m,
                         A_fm=prep.A_fm, At_fm=prep.At_fm,
                         log_weight_sum=st.log_weight_sum,
                         within_ss=prep.within_ss,
                         within_df=prep.within_df,
                         rho_eps=st.rho_eps)
    end

    # Starting values: ρ=0.5, σa ≈ 0.7*y_std, σz ≈ 0.04*y_std, σe ≈ 0.4*y_std
    # (informed by typical estimates from previous runs)
    p0 = [atanh(0.5/0.99), log(0.7), log(0.04), log(0.4)]
    if prep.rho_eps_estimate
        push!(p0, rhoeps_to_unconstrained(0.5))
    end

    @printf("\n--- Optimization (two-phase: L-BFGS then Nelder-Mead) ---\n")
    if length(p0) == 5
        @printf("Initial p0 = (%.6g, %.6g, %.6g, %.6g, %.6g)\n",
                p0[1], p0[2], p0[3], p0[4], p0[5])
    else
        @printf("Initial p0 = (%.6g, %.6g, %.6g, %.6g)\n", p0[1], p0[2], p0[3], p0[4])
    end

    # Phase 1: L-BFGS with finite differences for fast initial descent.
    # Cap at 20 iterations — FD gradient noise makes further L-BFGS
    # iterations useless (gradient noise floor ~ √ε × NLL ≈ O(1)).
    @printf("Phase 1: L-BFGS (max 20 iterations)...\n")
    phase1_opts = Optim.Options(
        iterations=20,
        show_trace=verbose,
        show_every=5,
        g_tol=0.0,  # don't stop on gradient — it's noisy
    )
    res1 = nothing
    t_phase1 = @elapsed begin
        # Optim changed autodiff API from Symbol to ADTypes.AbstractADType.
        # Try the old API first, then fallback to ADTypes, then Optim default.
        try
            res1 = optimize(obj, p0, LBFGS(), phase1_opts; autodiff=:finite)
        catch err
            msg = sprint(showerror, err)
            if err isa TypeError && occursin("keyword argument autodiff", msg)
                if verbose
                    @printf("autodiff=:finite not supported by this Optim version; retrying...\n")
                end
                try
                    res1 = optimize(obj, p0, LBFGS(), phase1_opts;
                                    autodiff=ADTypes.AutoFiniteDiff())
                catch err2
                    if verbose
                        @printf("ADTypes finite-diff fallback failed; using Optim default autodiff.\n")
                    end
                    res1 = optimize(obj, p0, LBFGS(), phase1_opts)
                end
            else
                rethrow(err)
            end
        end
    end
    p1 = Optim.minimizer(res1)
    if verbose
        ρ1 = 0.99*tanh(p1[1]); σa1 = exp(p1[2])*prep.y_std
        σz1 = exp(p1[3])*prep.y_std; σe1 = exp(p1[4])*prep.y_std
        if length(p1) == 5
            @printf("Phase 1 done: %d iters, %.1fs | ρ=%.4f σa=%.4f σz=%.4f σε=%.4f rho_eps=%.4f\n",
                    Optim.iterations(res1), t_phase1, ρ1, σa1, σz1, σe1,
                    rhoeps_from_unconstrained(p1[5]))
        else
            @printf("Phase 1 done: %d iters, %.1fs | ρ=%.4f σa=%.4f σz=%.4f σε=%.4f\n",
                    Optim.iterations(res1), t_phase1, ρ1, σa1, σz1, σe1)
        end
    end

    # Phase 2: Nelder-Mead for noise-robust polishing.
    # No gradients needed — works well with exact (deterministic) objective.
    # Cap at 200 iterations — NM converges the simplex quickly from a
    # good starting point, and further iterations are wasted once the
    # simplex diameter is at machine precision.
    nm_iters = min(iters, 400)
    @printf("Phase 2: Nelder-Mead (max %d iterations)...\n", nm_iters)
    res = nothing
    t_phase2 = @elapsed begin
        res = optimize(
            obj, p1, NelderMead(),
            Optim.Options(
                iterations=nm_iters,
                show_trace=verbose,
                show_every=50,
            )
        )
    end

    t_opt = t_phase1 + t_phase2
    p̂ = Optim.minimizer(res)

    ρ̂  = 0.99 * tanh(p̂[1])
    σâ = exp(p̂[2]) * prep.y_std
    σẑ = exp(p̂[3]) * prep.y_std
    σê = exp(p̂[4]) * prep.y_std
    rho_eps_hat = prep.rho_eps_estimate ? rhoeps_from_unconstrained(p̂[5]) : prep.rho_eps_likelihood

    nll_hat = obj(p̂)

    # Convergence check: re-evaluate at a small perturbation of each parameter.
    # If the NLL doesn't improve for any perturbation, we're at a minimum.
    δ = 1e-4
    converged = true
    for i in 1:length(p̂)
        p_plus = copy(p̂); p_plus[i] += δ
        p_minus = copy(p̂); p_minus[i] -= δ
        if obj(p_plus) < nll_hat - 1e-6 || obj(p_minus) < nll_hat - 1e-6
            converged = false
            break
        end
    end

    @printf("\n--- Done ---\n")
    @printf("Phase 1 (L-BFGS): %d iters, %.1fs\n", Optim.iterations(res1), t_phase1)
    @printf("Phase 2 (Nelder-Mead): %d iters, %.1fs\n", Optim.iterations(res), t_phase2)
    @printf("Converged: %s | obj evals: %d | total time: %.1fs\n",
            string(converged), eval_counter[], t_opt)
    @printf("NLL (scaled): %.6f\n", nll_hat)
    @printf("Estimates: rho=%.6f, sigma_a=%.6f, sigma_z=%.6f, sigma_eps=%.6f\n",
            ρ̂, σâ, σẑ, σê)
    if prep.obs_weighting == :effective
        @printf("Residual correlation: rho_eps=%.6f (%s)\n",
                rho_eps_hat, prep.rho_eps_estimate ? "estimated" : "fixed")
    end

    prep_final = prep
    if prep.rho_eps_estimate
        final_stats = build_match_weight_stats(
            prep.base_f_rows, prep.base_m_cols, prep.base_y, prep.base_T,
            prep.N_F, prep.N_M, rho_eps_hat
        )
        prep_final = merge(prep, (
            ydot=final_stats.ydot,
            projected_y=final_stats.projected_y,
            VtV=final_stats.VtV,
            cnt_f=final_stats.cnt_f,
            cnt_m=final_stats.cnt_m,
            A_obs=final_stats.A_obs,
            At_obs=final_stats.At_obs,
            log_weight_sum=final_stats.log_weight_sum,
            effective_weight_sum=final_stats.effective_weight_sum,
            effective_weight_over_T_sum=final_stats.effective_weight_over_T_sum,
            mean_effective_weight=final_stats.mean_effective_weight,
            max_effective_weight=final_stats.max_effective_weight,
            rho_eps_likelihood=rho_eps_hat
        ))
    end

    decomp = nothing
    post_decomp = nothing
    if decomp_probes > 0
        decomp = exact_variance_decomp(prep_final, p̂; target=decomp_target,
                                       m=decomp_probes, seed=seed, verbose=verbose,
                                       a_weighting=a_weighting)
        post_decomp = exact_posterior_decomp(prep_final, p̂; target=decomp_target,
                                             m=decomp_probes, seed=seed,
                                             verbose=verbose,
                                             a_weighting=a_weighting)
    end

    return (ok=true,
            outcome=outcome, seed=seed,
            a_weighting=a_weighting, prior_adjacency=prior_adjacency,
            decomp_target=decomp_target,
            obs_weighting=prep_final.obs_weighting,
            rho_eps_fixed=prep_final.rho_eps_fixed,
            rho_eps_estimate=prep_final.rho_eps_estimate,
            rho_eps=rho_eps_hat,
            personyear_rows=prep_final.personyear_rows,
            within_ss=prep_final.within_ss,
            within_df=prep_final.within_df,
            log_weight_sum=prep_final.log_weight_sum,
            effective_weight_sum=prep_final.effective_weight_sum,
            effective_weight_over_T_sum=prep_final.effective_weight_over_T_sum,
            mean_effective_weight=prep_final.mean_effective_weight,
            max_effective_weight=prep_final.max_effective_weight,
            N_F=prep_final.N_F, N_M=prep_final.N_M, K=prep_final.K, n=n,
            y_std=prep_final.y_std,
            unique_edges=prep_final.unique_edges, duplicate_rows=prep_final.duplicate_rows,
            mean_edge_count=prep_final.mean_edge_count, max_edge_count=prep_final.max_edge_count,
            total_prior_weight=prep_final.total_prior_weight,
            max_prior_degree_f=prep_final.max_prior_degree_f,
            max_prior_degree_m=prep_final.max_prior_degree_m,
            converged=converged, iterations=Optim.iterations(res),
            obj_evals=eval_counter[], opt_time=t_opt,
            nll=nll_hat, rho=ρ̂, sigma_a=σâ, sigma_z=σẑ, sigma_eps=σê,
            decomp=decomp, post_decomp=post_decomp)
end

# ============================================================
# 5) Path utilities
# ============================================================

"""
Parse input path and construct output path.
Supports two formats:
  temp/chunks/<chunk>/edgelist.parquet  -> output/gmrfmle/<chunk>/exact/estimates.txt
  temp/samples/<chunk>/<sample>/edgelist.parquet -> output/gmrfmle/<chunk>/<sample>/estimates.txt
Nested sample paths are kept in the output path after the first chunk segment.
"""
function parse_input_path(in_path::String)::Tuple{String,String,String}
    normalized = replace(in_path, "\\" => "/")
    parts = split(normalized, '/')

    # Try temp/samples/<chunk>/<sample>/... format first
    samples_idx = findfirst(==("samples"), parts)
    if samples_idx !== nothing && length(parts) >= samples_idx + 3
        chunk = parts[samples_idx + 1]
        sample_parts = parts[samples_idx + 2 : end - 1]
        sample = join(sample_parts, "/")
        out_path = joinpath("output", "gmrfmle", chunk, sample_parts..., "estimates.txt")
        return (chunk, sample, out_path)
    end

    # Try temp/chunks/<chunk>/... format
    chunks_idx = findfirst(==("chunks"), parts)
    if chunks_idx !== nothing && length(parts) >= chunks_idx + 2
        chunk_parts = parts[chunks_idx + 1 : end - 1]
        chunk = join(chunk_parts, "/")
        out_path = joinpath("output", "gmrfmle", chunk_parts..., "exact", "estimates.txt")
        return (chunk, "exact", out_path)
    end

    error("Cannot parse input path. Expected temp/chunks/<chunk>/... or temp/samples/<chunk>/<sample>/...: $in_path")
end

# ============================================================
# 6) Main
# ============================================================

function main()
    if length(ARGS) < 1
        println(stderr,
            "Usage:\n" *
            "  julia --project=. src/estimate/gmrfmle_exact.jl <path/to/edgelist.parquet> [outcome] [flags]\n" *
            "\n" *
            "Flags:\n" *
            "  --decompose      Run variance decomposition with 200 probes (default)\n" *
            "  --decompose=N    Run variance decomposition with N probes\n" *
            "  --no-decompose   Skip variance decomposition\n" *
            "  --a_weighting=<m>  Adjacency weighting: degree (default), unweighted\n" *
            "  --prior_adjacency=<m>  Prior graph adjacency: binary (default), counts\n" *
            "  --obs_weighting=<m>  Observation model: raw (default), edge, effective\n" *
            "  --rho_eps=<v|estimate>  Within-match residual correlation for effective weighting\n" *
            "  --decomp_target=<m>  Variance target: estimation (default), personyear, edge\n" *
            "\n" *
            "Input path formats:\n" *
            "  temp/chunks/<chunk>/edgelist.parquet\n" *
            "  temp/samples/<chunk>/<sample>/edgelist.parquet\n" *
            "\n" *
            "Output:\n" *
            "  output/gmrfmle/<chunk>/exact/estimates.txt\n" *
            "  output/gmrfmle/<chunk>/<sample>/estimates.txt\n" *
            "\n" *
            "Examples:\n" *
            "  julia src/estimate/gmrfmle_exact.jl temp/chunks/full/edgelist.parquet\n" *
            "  julia src/estimate/gmrfmle_exact.jl temp/chunks/1990-1995/edgelist.parquet lnR\n" *
            "  julia src/estimate/gmrfmle_exact.jl temp/chunks/personyear/full/edgelist.parquet lnR --prior_adjacency=counts\n" *
            "  julia src/estimate/gmrfmle_exact.jl temp/samples/personyear/full/minedge-5/edgelist.parquet lnR --obs_weighting=effective --rho_eps=estimate\n"
        )
        return 1
    end

    in_path = ARGS[1]
    outcome = (length(ARGS) >= 2 && !startswith(ARGS[2], "--")) ? Symbol(ARGS[2]) : :lnR

    # Parse flags
    decomp_probes = 200   # default: run decomposition
    a_weighting   = :degree
    prior_adjacency = :binary
    obs_weighting = :raw
    rho_eps_fixed = nothing
    rho_eps_estimate = false
    decomp_target = :estimation
    for arg in ARGS
        if arg == "--no-decompose"
            decomp_probes = 0
        elseif arg == "--decompose"
            decomp_probes = 200
        elseif startswith(arg, "--decompose=")
            decomp_probes = parse(Int, arg[length("--decompose=")+1:end])
        elseif startswith(arg, "--variance-target=")
            variance_target = Symbol(split(arg, '=', limit=2)[2])
            variance_target in (:structural, :akm) ||
                error("--variance-target must be structural or akm; got $variance_target")
            @warn "--variance-target is deprecated and ignored; gmrfmle_exact.jl now computes both prior and posterior variance decompositions whenever decomposition is enabled."
        elseif startswith(arg, "--a_weighting=")
            a_weighting = Symbol(split(arg, '=', limit=2)[2])
            a_weighting in (:degree, :unweighted) ||
                error("--a_weighting must be degree or unweighted; got $a_weighting")
        elseif startswith(arg, "--prior_adjacency=")
            prior_adjacency = Symbol(split(arg, '=', limit=2)[2])
            prior_adjacency in (:binary, :counts) ||
                error("--prior_adjacency must be binary or counts; got $prior_adjacency")
        elseif startswith(arg, "--obs_weighting=")
            obs_weighting = Symbol(split(arg, '=', limit=2)[2])
            obs_weighting in (:raw, :edge, :effective) ||
                error("--obs_weighting must be raw, edge, or effective; got $obs_weighting")
        elseif startswith(arg, "--rho_eps=")
            val = split(arg, '=', limit=2)[2]
            if lowercase(strip(val)) == "estimate"
                rho_eps_estimate = true
                rho_eps_fixed = nothing
            else
                rho_eps_fixed = parse(Float64, val)
                0.0 <= rho_eps_fixed < 1.0 ||
                    error("--rho_eps must satisfy 0 <= rho_eps < 1, got $rho_eps_fixed")
                rho_eps_estimate = false
            end
        elseif startswith(arg, "--decomp_target=")
            decomp_target = normalize_decomp_target(Symbol(split(arg, '=', limit=2)[2]))
        end
    end
    if obs_weighting == :effective && !rho_eps_estimate && rho_eps_fixed === nothing
        rho_eps_estimate = true
    end
    if obs_weighting != :effective && (rho_eps_estimate || rho_eps_fixed !== nothing)
        error("--rho_eps can only be used with --obs_weighting=effective")
    end

    isfile(in_path) || error("Input file not found: $in_path")

    chunk, sample_id, out_path = parse_input_path(in_path)
    suffixes = String[]
    if a_weighting != :degree
        push!(suffixes, String(a_weighting))
    end
    if prior_adjacency != :binary
        push!(suffixes, "prior-$(String(prior_adjacency))")
    end
    if obs_weighting != :raw
        push!(suffixes, "obs-$(String(obs_weighting))")
    end
    if obs_weighting == :effective
        if rho_eps_estimate
            push!(suffixes, "rhoeps-est")
        elseif rho_eps_fixed !== nothing
            push!(suffixes, "rhoeps$(safe_float_id(rho_eps_fixed))")
        end
    end
    if decomp_target != :estimation
        push!(suffixes, "decomp-$(String(decomp_target))")
    end
    if !isempty(suffixes)
        out_path = replace(out_path, "estimates.txt" => "estimates_$(join(suffixes, "_")).txt")
    end
    out_dir = dirname(out_path)
    mkpath(out_dir)

    @printf("Reading: %s\n", in_path)
    df = Parquet2.readfile(in_path) |> DataFrame
    @printf("Loaded rows: %d\n", nrow(df))

    r = estimate_exact(
        df;
        outcome=outcome,
        iters=1000,
        verbose=true,
        decomp_probes=decomp_probes,
        a_weighting=a_weighting,
        prior_adjacency=prior_adjacency,
        obs_weighting=obs_weighting,
        rho_eps_fixed=rho_eps_fixed,
        rho_eps_estimate=rho_eps_estimate,
        decomp_target=decomp_target,
        seed=20260126
    )

    open(out_path, "w") do io
        @printf(io, "GMRF MLE Estimation (exact sparse Cholesky + L-BFGS)\n")
        @printf(io, "Chunk=%s | Sample=%s\n", chunk, sample_id)
        @printf(io, "Input=%s\n", in_path)
        @printf(io, "Outcome=%s\n", String(outcome))
        @printf(io, "AdjacencyWeighting=%s\n", String(a_weighting))
        @printf(io, "PriorAdjacency=%s\n", String(r.prior_adjacency))
        @printf(io, "ObsWeighting=%s\n", String(r.obs_weighting))
        @printf(io, "DecompTarget=%s\n", String(r.decomp_target))
        if r.obs_weighting == :effective
            @printf(io, "RhoEpsMode=%s\n", r.rho_eps_estimate ? "estimated" : "fixed")
            @printf(io, "RhoEps=%.6f\n", r.rho_eps)
            @printf(io, "EffectiveWeight=r(T)=T/(1+rho_eps*(T-1))\n")
            @printf(io, "PersonYearRowsFinite=%d | EffectiveWeightSum=%.6f | MeanEffectiveWeight=%.6f | MaxEffectiveWeight=%.6f\n",
                    r.personyear_rows, r.effective_weight_sum, r.mean_effective_weight, r.max_effective_weight)
            @printf(io, "EffectiveWeightOverTSum=%.6f\n", r.effective_weight_over_T_sum)
            @printf(io, "WithinDF=%d | WithinSSScaled=%.6f | LogWeightSum=%.6f\n",
                    r.within_df, r.within_ss, r.log_weight_sum)
        elseif r.obs_weighting == :edge
            @printf(io, "PersonYearRowsFinite=%d | collapsed to one equally weighted row per unique edge\n",
                    r.personyear_rows)
        end
        @printf(io, "UniqueEdges=%d | DuplicateRows=%d | MeanEdgeCount=%.6f | MaxEdgeCount=%.0f\n",
                r.unique_edges, r.duplicate_rows, r.mean_edge_count, r.max_edge_count)
        @printf(io, "TotalPriorWeight=%.0f | MaxPriorDegreeFirm=%.0f | MaxPriorDegreeManager=%.0f\n",
                r.total_prior_weight, r.max_prior_degree_f, r.max_prior_degree_m)
        @printf(io, "Seed=%d\n", r.seed)
        @printf(io, "N_F=%d, N_M=%d, K=%d, n=%d\n", r.N_F, r.N_M, r.K, r.n)
        @printf(io, "y_std=%.6g (scaling factor)\n", r.y_std)
        @printf(io, "Converged: %s | Iterations: %d | Obj evals: %d | Time: %.1fs\n",
                string(r.converged), r.iterations, r.obj_evals, r.opt_time)
        @printf(io, "NLL (scaled): %.6f\n", r.nll)
        @printf(io, "Estimates (structural units):\n")
        @printf(io, "  rho        = %.6f\n", r.rho)
        @printf(io, "  sigma_a    = %.6f\n", r.sigma_a)
        @printf(io, "  sigma_z    = %.6f\n", r.sigma_z)
        @printf(io, "  sigma_eps  = %.6f\n", r.sigma_eps)
        if r.obs_weighting == :effective
            @printf(io, "  rho_eps    = %.6f  (%s within-match residual correlation)\n",
                    r.rho_eps, r.rho_eps_estimate ? "estimated" : "fixed")
        end
        if r.decomp !== nothing
            d = r.decomp
            @printf(io, "\nPrior Variance Decomposition (Object 2; target=%s, exact solves, probes=%d):\n",
                    String(d.target), d.probes)
            @printf(io, "  TargetWeightSum = %.6f | ResidualLevel=%s | ObservedSecondMoment=%.6f\n",
                    d.weight_sum, String(d.residual_level), d.observed_second_moment)
            @printf(io, "  V_firm     = %+.6f  (target-weighted firm effects)\n", d.V_firm)
            @printf(io, "  V_manager  = %+.6f  (target-weighted manager effects)\n", d.V_manager)
            @printf(io, "  V_cross    = %+.6f  (2 x sorting cov; positive when rho > 0)\n", d.V_cross)
            if d.residual_model == :compound_symmetric
                @printf(io, "  V_eta      = %+.6f  (persistent match-specific residual)\n", d.V_eta_match)
                @printf(io, "  V_u        = %+.6f  (target annual/idiosyncratic residual contribution)\n", d.V_u_target)
            end
            @printf(io, "  V_eps      = %+.6f  (target residual contribution)\n", d.V_eps)
            @printf(io, "  V_total    = %+.6f  (target observed second moment = %.6f)\n",
                    d.V_total, d.observed_second_moment)
            if d.V_total > 0
                @printf(io, "Variance shares:\n")
                @printf(io, "  Share_firm     = %+.4f\n", d.V_firm    / d.V_total)
                @printf(io, "  Share_manager  = %+.4f\n", d.V_manager / d.V_total)
                @printf(io, "  Share_cross    = %+.4f\n", d.V_cross   / d.V_total)
                @printf(io, "  Share_eps      = %+.4f\n", d.V_eps     / d.V_total)
            end
        end
        if r.post_decomp !== nothing
            p = r.post_decomp
            s = akm_comparable_summary(p)
            summary_label = (p.target == :personyear || p.residual_level == :annual) ?
                "AKM-Comparable Summary (posterior moments; directly comparable to AKM/KSS)" :
                "Target Posterior Summary (posterior moments on selected variance target)"
            @printf(io, "\n%s:\n", summary_label)
            @printf(io, "  rho        = %.6f\n", s.rho)
            @printf(io, "  sigma_a    = %.6f\n", s.sigma_a)
            @printf(io, "  sigma_z    = %.6f\n", s.sigma_z)
            @printf(io, "  sigma_eps  = %.6f\n", s.sigma_eps)
            @printf(io, "Variance decomposition (target=%s):\n", String(p.target))
            @printf(io, "  V_firm     = %+.6f\n", s.V_firm)
            @printf(io, "  V_manager  = %+.6f\n", s.V_manager)
            @printf(io, "  V_cross    = %+.6f\n", s.V_cross)
            @printf(io, "  V_eps      = %+.6f\n", s.V_eps)
            @printf(io, "  V_total    = %+.6f\n", s.V_total)
            @printf(io, "\nPosterior Variance Decomposition (Object 3; target=%s, exact solves, probes=%d):\n",
                    String(p.target), p.probes)
            @printf(io, "  TargetWeightSum = %.6f | ResidualLevel=%s | ObservedSecondMoment=%.6f\n",
                    p.weight_sum, String(p.residual_level), p.observed_second_moment)
            @printf(io, "  E[sigma_a^2|y]  = %+.6f  (quad=%+.6f, trace=%+.6f)\n",
                    p.E_sa2, p.E_sa2_quad, p.E_sa2_trace)
            @printf(io, "  E[sigma_z^2|y]  = %+.6f  (quad=%+.6f, trace=%+.6f)\n",
                    p.E_sz2, p.E_sz2_quad, p.E_sz2_trace)
            @printf(io, "  E[sigma_az |y]  = %+.6f  (quad=%+.6f, trace=%+.6f)\n",
                    p.E_saz, p.E_saz_quad, p.E_saz_trace)
            if p.residual_model == :compound_symmetric
                @printf(io, "  V_eta           = %+.6f  (persistent match-specific residual)\n", p.V_eta_match)
                @printf(io, "  V_u             = %+.6f  (target annual/idiosyncratic residual contribution)\n", p.V_u_target)
            end
            @printf(io, "  V_eps           = %+.6f  (target residual contribution)\n", p.V_eps)
            @printf(io, "  E_total         = %+.6f  (target observed second moment = %.6f)\n",
                    p.E_total, p.observed_second_moment)
            if p.E_total > 0
                @printf(io, "Posterior variance shares (2x sorting cov for cross):\n")
                @printf(io, "  Share_firm     = %+.4f\n", p.E_sa2          / p.E_total)
                @printf(io, "  Share_manager  = %+.4f\n", p.E_sz2          / p.E_total)
                @printf(io, "  Share_cross    = %+.4f\n", 2.0 * p.E_saz   / p.E_total)
                @printf(io, "  Share_eps      = %+.4f\n", p.V_eps          / p.E_total)
            end
        end
    end

    @printf("Wrote: %s\n", out_path)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
