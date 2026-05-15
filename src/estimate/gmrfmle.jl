#!/usr/bin/env julia
# ============================================================
# gmrfmle.jl
#
# GMRF Maximum Likelihood Estimation using Hutch/SLQ + PCG,
# with optional Hutchinson average marginal variance decomposition.
#
# Usage:
#   julia --project=. src/estimate/gmrfmle.jl <path> [outcome] [flags]
#
# Flags:
#   --decompose      Run variance decomposition with 200 Hutchinson probes (default)
#   --decompose=N    Run variance decomposition with N probes
#   --no-decompose   Skip variance decomposition
#   --fix-rho=<v>    Fix rho to value v (abs(v) < 0.99); optimize over sigma_a, sigma_z, sigma_eps only
#   --maxdeg=N       Remove firms/managers with degree > N before estimation
#   --a_weighting=<mode>  Adjacency weighting: degree (default), spectral, unweighted
#   --prior_adjacency=<mode>  Prior graph adjacency: binary (default), counts
#   --obs_weighting=<mode>  Observation model: raw (default), edge, effective
#   --rho_eps=<v|estimate>  Within-match residual correlation for effective weighting
#   --decomp_target=<mode>  Variance target: estimation (default), personyear, edge
#
# Input:
#   temp/samples/<chunk>/<sample>/edgelist.parquet
#
# Output:
#   output/gmrfmle/<chunk>/<sample>/estimates.txt          (default)
#   output/gmrfmle-rho<v>/<chunk>/<sample>/estimates.txt  (with --fix-rho=<v>)
#   output/gmrfmle-<mode>[-prior-counts][-maxdeg<N>][-obs-...]/...  (with non-default flags)
# ============================================================

const GMRFMLE_PROJECT_ROOT = realpath(normpath(joinpath(@__DIR__, "..", "..")))

using Parquet2
using DataFrames
using Optim
using FiniteDiff
using SparseArrays
using LinearAlgebra
using Statistics
using Random
using Printf: @printf, @sprintf

BLAS.set_num_threads(1)

to_float_nan(x) = x isa Missing ? NaN : Float64(x)

# Helper to make filesystem-safe IDs
safe_id(s::AbstractString) = replace(s, ":" => "_")

# ============================================================
# 1) Sufficient statistics for V
# ============================================================

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

    VtV = [spdiagm(0 => cnt_f)  A_obs;
           A_obs'              spdiagm(0 => cnt_m)]

    return projected_y, VtV, cnt_f, cnt_m, A_obs
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

    VtV = [spdiagm(0 => cnt_f)  A_obs;
           A_obs'              spdiagm(0 => cnt_m)]

    return projected_y, VtV, cnt_f, cnt_m, A_obs, ydot
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
    projected_y, VtV, cnt_f, cnt_m, A_obs, ydot =
        build_weighted_V_stats(f_rows, m_cols, y, w, N_f, N_m)
    return (projected_y=projected_y, VtV=VtV, cnt_f=cnt_f, cnt_m=cnt_m,
            A_obs=A_obs, At_obs=transpose(A_obs), ydot=ydot,
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
    projected_y, VtV, cnt_f, cnt_m, A_obs, ydot_mean =
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
            cnt_f=cnt_f, cnt_m=cnt_m, A_obs=A_obs, At_obs=transpose(A_obs),
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

    # Without rho_eps the fitted observation model has one residual variance.
    # For raw person-year input, an edge-mean target averages iid annual residuals.
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
# 2) PCG with workspace
# ============================================================

mutable struct PCGWorkspace
    x::Vector{Float64}
    r::Vector{Float64}
    z::Vector{Float64}
    p::Vector{Float64}
    Ap::Vector{Float64}
end
PCGWorkspace(n::Int) = PCGWorkspace(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n))

function pcg_solve!(
    ws::PCGWorkspace,
    mulA!,
    b::AbstractVector{<:Real};
    tol::Real=1e-6,
    maxiter::Int=700,
    Mdiag::Union{Nothing,AbstractVector{<:Real}}=nothing,
    diag_floor::Real=1e-12
)
    x, r, z, p, Ap = ws.x, ws.r, ws.z, ws.p, ws.Ap
    n = length(b)

    fill!(x, 0.0)
    @inbounds for i in 1:n
        r[i] = Float64(b[i])
    end

    nb = norm(r)
    if nb == 0.0
        return x, true, 0, 0.0
    end

    if Mdiag === nothing
        z .= r
    else
        @inbounds for i in 1:n
            d = Float64(Mdiag[i])
            d = (d > diag_floor) ? d : diag_floor
            z[i] = r[i] / d
        end
    end

    p .= z
    rz = dot(r, z)
    if !isfinite(rz)
        return x, false, 0, Inf
    end

    for it in 1:maxiter
        mulA!(Ap, p)
        denom = dot(p, Ap)
        if !(isfinite(denom) && denom > 0)
            return x, false, it, Inf
        end

        α = rz / denom
        if !isfinite(α)
            return x, false, it, Inf
        end

        @inbounds @simd for i in 1:n
            x[i] += α * p[i]
            r[i] -= α * Ap[i]
        end

        relres = norm(r) / nb
        if relres <= tol
            return x, true, it, relres
        end

        if Mdiag === nothing
            z .= r
        else
            @inbounds for i in 1:n
                d = Float64(Mdiag[i])
                d = (d > diag_floor) ? d : diag_floor
                z[i] = r[i] / d
            end
        end

        rz_new = dot(r, z)
        if !(isfinite(rz_new) && rz != 0.0)
            return x, false, it, Inf
        end

        β = rz_new / rz
        if !isfinite(β)
            return x, false, it, Inf
        end

        @inbounds @simd for i in 1:n
            p[i] = z[i] + β * p[i]
        end
        rz = rz_new
    end

    return x, false, maxiter, norm(r) / nb
end

# ============================================================
# 3) SLQ logdet with workspace
# ============================================================

mutable struct SLQWorkspace
    z::Vector{Float64}
    q::Vector{Float64}
    q0::Vector{Float64}
    w::Vector{Float64}
    α::Vector{Float64}
    β::Vector{Float64}
end
SLQWorkspace(n::Int, k::Int) = SLQWorkspace(
    Vector{Float64}(undef, n),
    zeros(n),
    zeros(n),
    zeros(n),
    zeros(k),
    zeros(max(k-1, 0))
)

function slq_logdet_spd_mul_cached!(
    mulA!,
    n::Int,
    ws::SLQWorkspace;
    m::Int=30,
    k::Int=30,
    seed::Int=1,
    jitter::Real=1e-12,
    ritz_floor::Real=1e-14
)
    k_eff = min(k, n)
    rng = MersenneTwister(seed)

    z, q, q0, w = ws.z, ws.q, ws.q0, ws.w
    αfull, βfull = ws.α, ws.β

    total = 0.0
    nz = sqrt(n)

    for _ in 1:m
        @inbounds for i in 1:n
            z[i] = rand(rng, Bool) ? 1.0 : -1.0
        end

        @. q = z / nz
        fill!(q0, 0.0)
        beta_prev = 0.0

        fill!(αfull, 0.0)
        fill!(βfull, 0.0)
        t_stop = k_eff

        for t in 1:k_eff
            mulA!(w, q)
            if jitter != 0
                @. w = w + jitter*q
            end

            αfull[t] = dot(q, w)
            @. w = w - αfull[t]*q - beta_prev*q0

            if t < k_eff
                beta_prev = norm(w)
                if !isfinite(beta_prev)
                    return NaN
                end
                βfull[t] = beta_prev

                if beta_prev < 1e-14
                    t_stop = t
                    break
                end

                q0 .= q
                @. q = w / beta_prev
            end
        end

        αv = copy(view(αfull, 1:t_stop))
        βv = (t_stop > 1) ? copy(view(βfull, 1:(t_stop-1))) : Float64[]

        T = SymTridiagonal(αv, βv)

        local E
        try
            E = eigen(T)
        catch
            return NaN
        end

        vals = E.values
        u1   = @view E.vectors[1, :]

        s = 0.0
        @inbounds for j in 1:length(vals)
            λj = vals[j] > ritz_floor ? vals[j] : ritz_floor
            s += log(λj) * (u1[j]^2)
        end

        total += (nz^2) * s
    end

    return total / m
end

# ============================================================
# 4a) Pre-estimation filters
# ============================================================

"""
    filter_maxdeg(df, maxdeg; verbose) -> DataFrame

Remove edges incident to firms or managers whose degree exceeds `maxdeg`.
Removing high-degree nodes can only decrease remaining nodes' degrees,
so a single pass suffices.  Any resulting isolated nodes are dropped
automatically when `prepare_data` re-indexes.
"""
function filter_maxdeg(df::DataFrame, maxdeg::Int; verbose::Bool=true)
    firm_deg   = combine(groupby(df, :frame_id_numeric), nrow => :deg)
    person_deg = combine(groupby(df, :person_id),        nrow => :deg)

    keep_firms   = Set(firm_deg[firm_deg.deg   .<= maxdeg, :frame_id_numeric])
    keep_persons = Set(person_deg[person_deg.deg .<= maxdeg, :person_id])

    n_before = nrow(df)
    df_out = filter(row -> row.frame_id_numeric in keep_firms &&
                           row.person_id in keep_persons, df)

    if verbose
        n_firms_dropped   = nrow(firm_deg)   - length(keep_firms)
        n_persons_dropped = nrow(person_deg) - length(keep_persons)
        @printf("maxdeg=%d: dropped %d firms, %d managers, %d→%d edges\n",
                maxdeg, n_firms_dropped, n_persons_dropped, n_before, nrow(df_out))
    end
    return df_out
end

# ============================================================
# 4b) Leading singular value (power iteration)
# ============================================================

"""
    leading_singular_value(A, At; maxiter, tol) -> Float64

Compute σ₁(A) = √λ_max(AᵀA) via power iteration.  No extra dependencies.
"""
function leading_singular_value(
    A::SparseMatrixCSC, At::Transpose{Float64,<:SparseMatrixCSC};
    maxiter::Int=200, tol::Float64=1e-10
)
    m = size(A, 2)
    rng = MersenneTwister(12345)
    x = randn(rng, m)
    x ./= norm(x)
    σ_old = 0.0
    tmp = Vector{Float64}(undef, size(A, 1))
    for _ in 1:maxiter
        mul!(tmp, A, x)         # tmp = A x
        mul!(x, At, tmp)        # x   = Aᵀ A x
        λ = norm(x)
        x ./= λ
        σ_new = sqrt(λ)
        if abs(σ_new - σ_old) / max(σ_new, 1e-15) < tol
            return σ_new
        end
        σ_old = σ_new
    end
    return σ_old
end

# ============================================================
# 4c) Prepare data
# ============================================================

function prepare_data(
    df::DataFrame;
    outcome::Symbol,
    a_weighting::Symbol=:degree,
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

    if nrow(df) == 0
        error("Empty dataset.")
    end

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

    if length(y_raw) == 0
        error("No usable observations after filtering non-missing $(outcome).")
    end

    μy = mean(y_raw)
    y_std = std(y_raw)
    if !(isfinite(y_std) && y_std > 0)
        error("Outcome has zero/invalid std. Cannot scale.")
    end

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

        projected_y, VtV, cnt_f, cnt_m, A_obs = build_V_stats(f_rows, m_cols, y, N_F, N_M)
        At_obs = transpose(A_obs)
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
            projected_y, VtV, cnt_f, cnt_m, A_obs, ydot =
                build_weighted_V_stats(f_rows, m_cols, y, w, N_F, N_M)
            At_obs = transpose(A_obs)
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
            within_ss = sum(Float64.(collapsed.ssw_raw)) / (y_std^2)
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
        @printf("Building V stats and sparse structures...\n")
    end

    prior_adjacency in (:binary, :counts) ||
        error("Unknown prior_adjacency: $(prior_adjacency). Use :binary or :counts.")

    A_fm = copy(A_prior_base)
    if prior_adjacency == :binary
        A_fm.nzval .= 1.0
    end
    At_fm = transpose(A_fm)

    d_f = vec(sum(A_fm; dims=2))
    d_m = vec(sum(A_fm; dims=1))
    if any(d_f .<= 0) || any(d_m .<= 0)
        error("Zero-degree node detected; investigate IDs / data construction.")
    end

    if verbose
        @printf("Prior adjacency: %s | unique_edges=%d duplicate_rows=%d mean_edge_count=%.3f max_edge_count=%.0f\n",
                String(prior_adjacency), unique_edges, duplicate_rows, mean_edge_count, max_edge_count)
        @printf("Prior graph degrees: max_firm=%.0f max_manager=%.0f total_prior_weight=%.0f\n",
                maximum(d_f), maximum(d_m), sum(A_fm.nzval))
    end

    # Adjacency weighting — controls the GMRF precision matrix Q.
    #
    # QOp implements the block structure:
    #   Q = [ diag(dw_f)/σ²_a      −(ρ/σ_aσ_z) W  ]
    #       [ −(ρ/σ_aσ_z) Wᵀ    diag(dw_m)/σ²_z  ]
    # where W = diag(df_is) * A * diag(dm_is).
    #
    # All three modes are PD for |ρ| < 1:
    #
    #   degree:     Q = S⁻¹(I − ρ D⁻¹ᐟ²AD⁻¹ᐟ²)S⁻¹        (normalized Laplacian)
    #               dw = 1, df_is = D_f^{-1/2}, dm_is = D_m^{-1/2}
    #               Eigenvalues of D^{-1/2}AD^{-1/2} in [-1,1] for bipartite graphs.
    #
    #   spectral:   Q = S⁻¹(I − ρ A/σ₁(A))S⁻¹               (spectral normalization)
    #               dw = 1, df_is = dm_is = 1/√σ₁(A)
    #               opnorm(A/σ���) = 1 by construction.
    #
    #   unweighted: Q = S⁻¹(D − ρ A)S⁻¹                     (unnormalized / paper)
    #               dw = d, df_is = dm_is = 1
    #               D − ρA = D¹ᐟ²(I − ρ D⁻¹ᐟ²AD⁻¹ᐟ²)D¹ᐟ² is PD by congruence.
    #
    # See docs/2026-03-19-Q-revisited.md for the algebraic relationship.
    if a_weighting == :degree
        df_is = 1.0 ./ sqrt.(d_f)
        dm_is = 1.0 ./ sqrt.(d_m)
        dw_f  = ones(Float64, N_F)
        dw_m  = ones(Float64, N_M)
    elseif a_weighting == :spectral
        s1 = leading_singular_value(A_fm, At_fm)
        if verbose
            @printf("Spectral normalization: σ₁(A) = %.6g, W = A/σ₁(A)\n", s1)
        end
        df_is = fill(1.0 / sqrt(s1), N_F)
        dm_is = fill(1.0 / sqrt(s1), N_M)
        dw_f  = ones(Float64, N_F)
        dw_m  = ones(Float64, N_M)
    elseif a_weighting == :unweighted
        if verbose
            @printf("Unweighted (paper) formulation: Q = S^{-1}(D - rho*A)S^{-1}\n")
        end
        df_is = ones(Float64, N_F)
        dm_is = ones(Float64, N_M)
        dw_f  = Float64.(d_f)
        dw_m  = Float64.(d_m)
    else
        error("Unknown a_weighting: $(a_weighting). Use :degree, :spectral, or :unweighted.")
    end

    return (y=y, ydot=ydot, y_std=y_std,
            A_fm=A_fm, At_fm=At_fm, df_is=df_is, dm_is=dm_is,
            dw_f=dw_f, dw_m=dw_m, d_f=d_f, d_m=d_m,
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
            cnt_f=cnt_f, cnt_m=cnt_m, A_obs=A_obs, At_obs=At_obs,
            projected_y=projected_y, VtV=VtV,
            firms=firms, people=people,
            base_f_rows=f_rows, base_m_cols=m_cols, base_y=y, base_T=T_edge,
            decomp_f_rows=decomp_f_rows, decomp_m_cols=decomp_m_cols,
            decomp_y=decomp_y, decomp_T=decomp_T,
            personyear_within_ss=personyear_within_ss,
            K=K, N_F=N_F, N_M=N_M)
end

# ============================================================
# 5) Callable operator structs
# ============================================================

mutable struct QOp{TA,TAT}
    A::TA
    At::TAT
    df_is::Vector{Float64}      # off-diagonal scaling (per-node)
    dm_is::Vector{Float64}      # off-diagonal scaling (per-node)
    dw_f::Vector{Float64}       # diagonal weights for firms  (ones or d_f)
    dw_m::Vector{Float64}       # diagonal weights for managers (ones or d_m)
    Nf::Int
    inv_sa2::Float64
    inv_sz2::Float64
    cross::Float64
    tmpF::Vector{Float64}
    tmpM::Vector{Float64}
end

"""
Apply Q x where Q has the block structure:

    Q = [ diag(dw_f)/σ²_a      −(ρ/σ_aσ_z) W  ]
        [ −(ρ/σ_aσ_z) Wᵀ    diag(dw_m)/σ²_z  ]

with W = diag(df_is) A diag(dm_is).

  degree:     dw = 1, df_is/dm_is = D^{-1/2}  →  Q = S⁻¹(I − ρ D⁻¹ᐟ²AD⁻¹ᐟ²)S⁻¹
  spectral:   dw = 1, df_is/dm_is = σ₁⁻¹ᐟ²   →  Q = S⁻¹(I − ρ A/σ₁)S⁻¹
  unweighted: dw = d, df_is/dm_is = 1          →  Q = S⁻¹(D − ρ A)S⁻¹
"""
function (op::QOp)(y::Vector{Float64}, x::Vector{Float64})
    n  = length(x)
    Nf = op.Nf
    @views xf = x[1:Nf]
    @views xm = x[(Nf+1):n]
    @views yf = y[1:Nf]
    @views ym = y[(Nf+1):n]

    inv_sa2 = op.inv_sa2
    inv_sz2 = op.inv_sz2
    cross   = op.cross

    # Diagonal blocks: diag(dw_f) / σ²_a  and  diag(dw_m) / σ²_z
    @. yf = inv_sa2 * op.dw_f * xf
    @. ym = inv_sz2 * op.dw_m * xm

    # Off-diagonal: −(ρ/σ_aσ_z) diag(df_is) A diag(dm_is)
    @. op.tmpM = op.dm_is * xm
    mul!(op.tmpF, op.A, op.tmpM)
    @. op.tmpF = op.df_is * op.tmpF
    @. yf -= cross * op.tmpF

    @. op.tmpF = op.df_is * xf
    mul!(op.tmpM, op.At, op.tmpF)
    @. op.tmpM = op.dm_is * op.tmpM
    @. ym -= cross * op.tmpM

    return y
end

mutable struct MOp{TQ,TV}
    qop::TQ
    VtV::TV
    tmpMV::Vector{Float64}
    λ::Float64
end

function (op::MOp)(y::Vector{Float64}, x::Vector{Float64})
    op.qop(y, x)
    mul!(op.tmpMV, op.VtV, x)
    @. y = y + op.λ * op.tmpMV
    return y
end

# ============================================================
# 6) Cached objective
# ============================================================

mutable struct NLLCache{TQ,TM}
    dV::Vector{Float64}
    Mdiag::Vector{Float64}
    pcg::PCGWorkspace
    slqQ::SLQWorkspace
    slqM::SLQWorkspace
    qop::TQ
    mop::TM
end

function nll_hutch_cached(
    params,
    ydot::Float64,
    projected_y::Vector{Float64},
    K::Int,
    cache::NLLCache;
    m::Int=30,
    k::Int=30,
    seed::Int=20260126,
    jitter::Real=1e-12,
    ritz_floor::Real=1e-14,
    cg_tol::Real=1e-6,
    cg_maxiter::Int=700,
    cg_diag_floor::Real=1e-12,
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

    if !(isfinite(ρ) && isfinite(σa) && isfinite(σz) && isfinite(σe)) || (σe <= 0) || (σa <= 0) || (σz <= 0)
        return BIG
    end

    λ = 1.0 / (σe^2)
    inv_sa2 = 1.0 / (σa^2)
    inv_sz2 = 1.0 / (σz^2)
    cross   = ρ / (σa * σz)

    n  = length(projected_y)
    Nf = cache.qop.Nf

    cache.qop.inv_sa2 = inv_sa2
    cache.qop.inv_sz2 = inv_sz2
    cache.qop.cross   = cross
    cache.mop.λ       = λ

    dV = cache.dV
    Mdiag = cache.Mdiag
    # Diagonal of Q (= dw * inv_s2) plus λ * diag(V'V).
    # For degree/spectral modes dw=1; for unweighted (paper) dw=d_i.
    dw_f = cache.qop.dw_f
    dw_m = cache.qop.dw_m
    @inbounds for i in 1:n
        dq = (i <= Nf) ? dw_f[i] * inv_sa2 : dw_m[i - Nf] * inv_sz2
        Mdiag[i] = dq + λ * dV[i]
    end

    xsol, ok, _, _ = pcg_solve!(cache.pcg, cache.mop, projected_y;
        tol=cg_tol, maxiter=cg_maxiter, Mdiag=Mdiag, diag_floor=cg_diag_floor
    )
    if !ok
        return BIG
    end

    quad_term = dot(projected_y, xsol)
    if !isfinite(quad_term)
        return BIG
    end

    ldQ = slq_logdet_spd_mul_cached!(cache.qop, n, cache.slqQ;
        m=m, k=k, seed=seed + 0, jitter=jitter, ritz_floor=ritz_floor
    )
    ldM = slq_logdet_spd_mul_cached!(cache.mop, n, cache.slqM;
        m=m, k=k, seed=seed + 10_000, jitter=jitter, ritz_floor=ritz_floor
    )
    if !(isfinite(ldQ) && isfinite(ldM))
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
        (K * (2.0*log(σe))) - Float64(log_weight_sum) +
        (ldM - ldQ) + (λ * ydot) - ((λ^2) * quad_term) +
        residual_corr_term
    )
    return isfinite(val) ? val : BIG
end

# ============================================================
# 6.5) Hutchinson average marginal variance decomposition
# ============================================================

"""
    hutch_variance_decomp(prep, p̂; m, seed, cg_tol, cg_maxiter, verbose)

Compute the prior-implied average marginal variance decomposition (Object 2 in
docs/2026-04-05-q-to-akm.md) via Hutchinson estimation:

    target^{-1} tr(Var(y_target)) = V_firm + V_manager + V_cross + V_eps

where

    V_firm    = (y_std² / W) · tr(diag(cnt_f(w)) · Σ_FF)
    V_manager = (y_std² / W) · tr(diag(cnt_m(w)) · Σ_MM)
    V_cross   = 2 · (y_std² / W) · tr(A_obs(w) · Σ_MF)
    V_eps     = target-specific residual contribution

The target weights w can be the likelihood weights, raw person-year exposure
weights, or one-per-edge weights. W = sum(w).

Q is the GMRF prior precision of the latent field x = (a, z), built at the
MLE estimates encoded in p̂ (the raw Nelder-Mead vector in scaled units).
Note: Q here is the *prior* precision, not the posterior precision (MOp).

In the code convention: the first N_F entries of x are *firm* effects
(parametrised by sigma_a), and the last N_M entries are *manager* effects
(parametrised by sigma_z).

V_cross = 2 × E_prior[sigma_az]. V_cross is positive when rho > 0.
"""
function hutch_variance_decomp(
    prep,
    p̂::Vector{Float64};
    target::Symbol=:estimation,
    m::Int=50,
    seed::Int=20260126,
    cg_tol::Float64=1e-6,
    cg_maxiter::Int=700,
    verbose::Bool=true
)
    # Reconstruct QOp at MLE estimates from raw optimizer vector (scaled units).
    # A fresh QOp is used here to avoid state left by the post-convergence
    # perturbation checks in estimate_hutch, which mutate the shared cache.qop.
    ρ  = 0.99 * tanh(p̂[1])
    σa = exp(p̂[2])
    σz = exp(p̂[3])
    σe = exp(p̂[4])

    inv_sa2 = 1.0 / σa^2
    inv_sz2 = 1.0 / σz^2
    cross   = ρ / (σa * σz)

    n = prep.N_F + prep.N_M

    qop = QOp(prep.A_fm, prep.At_fm, prep.df_is, prep.dm_is,
              prep.dw_f, prep.dw_m, prep.N_F,
              inv_sa2, inv_sz2, cross,
              zeros(prep.N_F), zeros(prep.N_M))

    pcg_ws = PCGWorkspace(n)
    rng    = MersenneTwister(seed + 77_777)  # independent from estimation probes

    dstats = decomp_target_stats(prep, target)
    cnt_f = dstats.cnt_f
    cnt_m = dstats.cnt_m

    acc_F     = 0.0
    acc_M     = 0.0
    acc_cross = 0.0
    n_ok      = 0

    v    = Vector{Float64}(undef, n)
    wv_f = Vector{Float64}(undef, prep.N_F)   # A_obs  * u_m
    wv_m = Vector{Float64}(undef, prep.N_M)   # At_obs * u_f

    verbose && @printf("\n--- Variance decomposition (target=%s, Hutchinson, m=%d) ---\n",
                       String(dstats.target), m)

    t_decomp = @elapsed for t in 1:m
        # Rademacher probe
        @inbounds for i in 1:n
            v[i] = rand(rng, Bool) ? 1.0 : -1.0
        end

        # Solve Q u = v  (prior precision, not posterior MOp)
        u, ok, _, relres = pcg_solve!(pcg_ws, qop, v; tol=cg_tol, maxiter=cg_maxiter)
        if ok
            n_ok += 1
        elseif verbose
            @printf("[decomp] probe %d: PCG did not converge (relres=%.2e)\n", t, relres)
        end

        vf = view(v, 1:prep.N_F)
        vm = view(v, prep.N_F+1:n)
        uf = view(u, 1:prep.N_F)
        um = view(u, prep.N_F+1:n)

        # Firm:    v' diag(cnt_f) u  =  sum_i cnt_f[i] * v[i] * u[i]
        @inbounds for i in 1:prep.N_F
            acc_F += cnt_f[i] * vf[i] * uf[i]
        end

        # Manager: v' diag(cnt_m) u  =  sum_j cnt_m[j] * v[N_F+j] * u[N_F+j]
        @inbounds for j in 1:prep.N_M
            acc_M += cnt_m[j] * vm[j] * um[j]
        end

        # Cross:   v' [0, A_obs; A_obs', 0] u  =  v_f'(A_obs u_m) + v_m'(A_obs' u_f)
        mul!(wv_f, dstats.A_obs,  um)
        mul!(wv_m, dstats.At_obs, uf)
        acc_cross += dot(vf, wv_f) + dot(vm, wv_m)
    end

    # Q is the precision of y/y_std, so Σ_x = Q⁻¹ is in scaled units.
    # Var(y_k) = y_std² · (Σ_x entries), hence multiply traces by y_std².
    scale     = prep.y_std^2 / (Float64(m) * dstats.weight_sum)
    V_firm    = acc_F     * scale
    V_manager = acc_M     * scale
    V_cross   = acc_cross * scale
    resid     = residual_decomp_components(prep, σe * prep.y_std, dstats)
    V_eps     = resid.V_eps_target
    V_total   = V_firm + V_manager + V_cross + V_eps

    if verbose
        @printf("Decomp time: %.1fs | PCG converged: %d/%d\n", t_decomp, n_ok, m)
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
            probes=m, pcg_converged=n_ok)
end

# ============================================================
# 6.6) Hutchinson posterior variance decomposition
# ============================================================

"""
    hutch_posterior_decomp(prep, p̂; m, seed, cg_tol, cg_maxiter, verbose)

Compute the posterior expected AKM variance decomposition (Object 3 in
docs/2026-04-05-q-to-akm.md):

    E[sigma_a^2  | y] = theta_hat' A_a  theta_hat + tr(A_a  Sigma_post)
    E[sigma_z^2  | y] = theta_hat' A_z  theta_hat + tr(A_z  Sigma_post)
    E[sigma_az   | y] = theta_hat' A_az theta_hat + tr(A_az Sigma_post)

where:
    Sigma_post = (Q + lambda * X'X)^{-1} = MOp^{-1}
    theta_hat  = MOp^{-1} * (lambda * projected_y)   (posterior mode)
    lambda     = 1 / sigma_eps^2   (in scaled units)
    A_a  = (1/W)  [diag(cnt_f(w)), 0 ; 0, 0]
    A_az = (1/2W)[0, A_obs(w); A_obs(w)', 0]

The quadratic terms (theta_hat' A theta_hat) capture the signal in the posterior
mode; the trace terms (tr(A Sigma_post)) capture residual posterior uncertainty.
"""
function hutch_posterior_decomp(
    prep,
    p̂::Vector{Float64};
    target::Symbol=:estimation,
    m::Int=50,
    seed::Int=20260126,
    cg_tol::Float64=1e-6,
    cg_maxiter::Int=700,
    verbose::Bool=true
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

    qop = QOp(prep.A_fm, prep.At_fm, prep.df_is, prep.dm_is,
              prep.dw_f, prep.dw_m, prep.N_F,
              inv_sa2, inv_sz2, cross,
              zeros(prep.N_F), zeros(prep.N_M))

    mop = MOp(qop, prep.VtV, zeros(n), λ)

    pcg_ws = PCGWorkspace(n)
    rng    = MersenneTwister(seed + 88_888)  # independent from prior decomp probes

    dstats = decomp_target_stats(prep, target)
    cnt_f = dstats.cnt_f
    cnt_m = dstats.cnt_m

    verbose && @printf("\n--- Posterior decomposition (target=%s, Hutchinson, m=%d) ---\n",
                       String(dstats.target), m)

    # ── Step 1: posterior mode  theta_hat = MOp^{-1} (lambda * projected_y) ────
    rhs = λ .* prep.projected_y
    theta_hat_ref, ok_mode, _, relres_mode = pcg_solve!(pcg_ws, mop, rhs;
        tol=cg_tol, maxiter=cg_maxiter)
    if !ok_mode
        verbose && @printf("[post_decomp] PCG for posterior mode did not converge (relres=%.2e)\n",
                           relres_mode)
        return nothing
    end
    # pcg_solve! returns ws.x by reference; copy before the probe loop overwrites it.
    theta_hat = copy(theta_hat_ref)
    th_f = view(theta_hat, 1:prep.N_F)
    th_m = view(theta_hat, prep.N_F+1:n)

    # ── Step 2: quadratic terms  theta_hat' A theta_hat  (scaled units) ────────
    # A_a  = (1/K) [diag(cnt_f), 0; 0, 0]  →  K * theta_hat' A_a  theta_hat = dot(cnt_f, th_f.^2)
    # A_z  = (1/K) [0, 0; 0, diag(cnt_m)]  →  K * theta_hat' A_z  theta_hat = dot(cnt_m, th_m.^2)
    # A_az = (1/2K)[0, A_obs; A_obs', 0]   →  K * theta_hat' A_az theta_hat = dot(th_f, A_obs*th_m)
    #   (A_az is symmetric, so th' A_az th = (1/2K)(th_f'A_obs th_m + th_m'A_obs'th_f)
    #    = (1/K) th_f' A_obs th_m  since scalar = its own transpose)
    tmp_Aobs_thm = Vector{Float64}(undef, prep.N_F)
    mul!(tmp_Aobs_thm, dstats.A_obs, th_m)

    qa  = dot(cnt_f, th_f .^ 2)          # K * theta_hat' A_a  theta_hat
    qz  = dot(cnt_m, th_m .^ 2)          # K * theta_hat' A_z  theta_hat
    qaz = dot(th_f, tmp_Aobs_thm)        # K * theta_hat' A_az theta_hat

    # ── Step 3: Hutchinson traces  tr(A Sigma_post)  via MOp solves ─────────────
    # For each probe v: solve MOp u = v, then:
    #   tr(diag(cnt_f) Sigma_post_FF) ≈ (1/m) sum_t vf' diag(cnt_f) uf
    #   tr(A_obs Sigma_post_MF)       ≈ (1/m) sum_t (vf' A_obs um + vm' A_obs' uf) / 2
    #
    # We accumulate:
    #   acc_F     ≈ m * tr(diag(cnt_f) Sigma_post_FF)   [firms only]
    #   acc_M     ≈ m * tr(diag(cnt_m) Sigma_post_MM)   [managers only]
    #   acc_cross ≈ m * tr([0,A_obs;A_obs',0] Sigma_post)  [= 2K * m * tr(A_az Sigma_post)]
    acc_F     = 0.0
    acc_M     = 0.0
    acc_cross = 0.0
    n_ok      = 0

    v    = Vector{Float64}(undef, n)
    wv_f = Vector{Float64}(undef, prep.N_F)   # A_obs  * u_m
    wv_m = Vector{Float64}(undef, prep.N_M)   # At_obs * u_f

    t_decomp = @elapsed for t in 1:m
        @inbounds for i in 1:n
            v[i] = rand(rng, Bool) ? 1.0 : -1.0
        end

        # Solve MOp u = v  (posterior precision = Q + lambda*VtV)
        u, ok, _, relres = pcg_solve!(pcg_ws, mop, v; tol=cg_tol, maxiter=cg_maxiter)
        if ok
            n_ok += 1
        elseif verbose
            @printf("[post_decomp] probe %d: PCG did not converge (relres=%.2e)\n", t, relres)
        end

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

    # ── Step 4: combine, unscale to original units ───────────────────────────────
    # quad_scale  converts K * (theta_hat' A theta_hat) → original-unit E[.|y] quad term
    # trace_scale converts m * tr(diag(cnt) Sigma_post) → original-unit trace term
    # For the cross: acc_cross estimates m * tr([0,A_obs;A_obs',0] Sigma_post)
    #   = 2Km * tr(A_az Sigma_post), so E_saz_trace = acc_cross * y_std^2 / (2*m*K).
    quad_scale   = prep.y_std^2 / dstats.weight_sum
    trace_scale  = prep.y_std^2 / (Float64(m) * dstats.weight_sum)

    E_sa2_quad   = qa  * quad_scale
    E_sa2_trace  = acc_F     * trace_scale
    E_sa2        = E_sa2_quad  + E_sa2_trace

    E_sz2_quad   = qz  * quad_scale
    E_sz2_trace  = acc_M     * trace_scale
    E_sz2        = E_sz2_quad  + E_sz2_trace

    E_saz_quad   = qaz * quad_scale
    E_saz_trace  = acc_cross * trace_scale / 2.0   # factor 1/2: acc_cross = 2*m*K*tr(A_az Sigma_post)/y_std^2
    E_saz        = E_saz_quad  + E_saz_trace

    resid  = residual_decomp_components(prep, σe * prep.y_std, dstats)
    V_eps  = resid.V_eps_target
    E_tot  = E_sa2 + E_sz2 + 2.0 * E_saz + V_eps

    if verbose
        @printf("Decomp time: %.1fs | PCG (traces) converged: %d/%d\n", t_decomp, n_ok, m)
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
            probes=m, pcg_converged_mode=ok_mode, pcg_converged_traces=n_ok)
end

# ============================================================
# 7) Estimation
# ============================================================

function estimate_hutch(
    df::DataFrame;
    outcome::Symbol=:lnR,
    a_weighting::Symbol=:degree,
    prior_adjacency::Symbol=:binary,
    obs_weighting::Symbol=:raw,
    rho_eps_fixed::Union{Float64,Nothing}=nothing,
    rho_eps_estimate::Bool=false,
    HUTCH_M::Int=8,
    HUTCH_K::Int=15,
    CG_TOL::Float64=1e-6,
    CG_MAXITER::Int=700,
    iters::Int=1000,
    seed::Int=20260126,
    verbose::Bool=true,
    decomp_probes::Int=0,
    decomp_target::Symbol=:estimation,
    rho_fixed::Union{Float64,Nothing}=nothing
)
    decomp_target = normalize_decomp_target(decomp_target)
    @printf("\n--- Preparing data ---\n")
    prep = prepare_data(df; outcome=outcome, a_weighting=a_weighting,
                        prior_adjacency=prior_adjacency,
                        obs_weighting=obs_weighting,
                        rho_eps_fixed=rho_eps_fixed,
                        rho_eps_estimate=rho_eps_estimate,
                        verbose=verbose)

    n = prep.N_F + prep.N_M
    @printf("\n--- Model dimensions ---\n")
    @printf("N_F=%d, N_M=%d, K=%d, n=%d\n", prep.N_F, prep.N_M, prep.K, n)
    @printf("HUTCH: m=%d, k=%d | PCG tol=%.2e, maxiter=%d | seed=%d\n",
            HUTCH_M, HUTCH_K, CG_TOL, CG_MAXITER, seed)

    dV = Vector{Float64}(diag(prep.VtV))

    qop = QOp(prep.A_fm, prep.At_fm, prep.df_is, prep.dm_is,
              prep.dw_f, prep.dw_m, prep.N_F,
              1.0, 1.0, 0.0,
              zeros(prep.N_F), zeros(prep.N_M))

    mop = MOp(qop, prep.VtV, zeros(n), 1.0)

    cache = NLLCache(
        dV,
        zeros(n),
        PCGWorkspace(n),
        SLQWorkspace(n, HUTCH_K),
        SLQWorkspace(n, HUTCH_K),
        qop,
        mop
    )

    eval_counter = Ref(0)

    function stats_for_objective!(p_full)
        if prep.rho_eps_estimate
            rho_eps_cur = rhoeps_from_unconstrained(p_full[5])
            stats = build_match_weight_stats(
                prep.base_f_rows, prep.base_m_cols, prep.base_y, prep.base_T,
                prep.N_F, prep.N_M, rho_eps_cur
            )
            cache.mop.VtV = stats.VtV
            cache.dV .= Vector{Float64}(diag(stats.VtV))
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
        if verbose && (eval_counter[] == 1 || eval_counter[] % 10 == 0)
            if length(p) == 5
                @printf("[obj] eval=%d | p=(%.3g, %.3g, %.3g, %.3g, %.3g)\n",
                        eval_counter[], p[1], p[2], p[3], p[4], p[5])
            else
                @printf("[obj] eval=%d | p=(%.3g, %.3g, %.3g, %.3g)\n",
                        eval_counter[], p[1], p[2], p[3], p[4])
            end
        end
        st = stats_for_objective!(p)
        return nll_hutch_cached(
            p, st.ydot, st.projected_y, prep.K, cache;
            m=HUTCH_M, k=HUTCH_K, seed=seed, jitter=1e-12,
            cg_tol=CG_TOL, cg_maxiter=CG_MAXITER,
            log_weight_sum=st.log_weight_sum,
            within_ss=prep.within_ss,
            within_df=prep.within_df,
            rho_eps=st.rho_eps
        )
    end

    @printf("\n--- Optimization (Nelder-Mead) ---\n")
    # Starting values informed by typical estimates from previous runs:
    # rho~0.5, sigma_a~0.7*y_std, sigma_z~0.04*y_std, sigma_e~0.4*y_std.
    # L-BFGS is not used here: each SLQ/PCG evaluation costs several seconds,
    # so FD-gradient computation (8 evals per L-BFGS step) is prohibitively slow.

    # g_tol sets the NM convergence criterion: nm_x = sqrt(var(f_simplex)*m/n) <= g_tol.
    # The default 1e-8 is designed for exact objective functions; the SLQ/PCG
    # approximation has intrinsic f-variation of ~1e-3 between nearby simplex vertices
    # when the GMRF precision matrix is ill-conditioned (e.g. small sigma_z).
    # Setting g_tol=1e-3 declares convergence when the f-spread matches the
    # approximation quality floor -- no parameter constraint, just honest bookkeeping.
    nm_opts = Optim.Options(iterations=iters, g_tol=1e-3, show_trace=verbose, show_every=50)
    delta   = 1e-4

    # Declare result variables in outer scope so both branches can assign them.
    res           = nothing
    t_opt         = 0.0
    ρ̂             = 0.0
    σâ            = 0.0
    σẑ            = 0.0
    σê            = 0.0
    rho_eps_hat   = prep.rho_eps_likelihood
    nll_hat       = 0.0
    converged     = false
    p̂_for_decomp  = zeros(prep.rho_eps_estimate ? 5 : 4)

    if rho_fixed === nothing
        # ── 4D unconstrained path ────────────────────────────────────────────────
        # Identical to the original estimation; rho is a free parameter.
        p0 = [atanh(0.5/0.99), log(0.7), log(0.04), log(0.4)]
        if prep.rho_eps_estimate
            push!(p0, rhoeps_to_unconstrained(0.5))
        end
        if length(p0) == 5
            @printf("Initial p0 = (%.6g, %.6g, %.6g, %.6g, %.6g)\n",
                    p0[1], p0[2], p0[3], p0[4], p0[5])
        else
            @printf("Initial p0 = (%.6g, %.6g, %.6g, %.6g)\n", p0[1], p0[2], p0[3], p0[4])
        end

        t_opt = @elapsed (res = optimize(obj, p0, NelderMead(), nm_opts))
        p̂ = Optim.minimizer(res)

        ρ̂  = 0.99 * tanh(p̂[1])
        σâ = exp(p̂[2]) * prep.y_std
        σẑ = exp(p̂[3]) * prep.y_std
        σê = exp(p̂[4]) * prep.y_std
        rho_eps_hat = prep.rho_eps_estimate ? rhoeps_from_unconstrained(p̂[5]) : prep.rho_eps_likelihood
        nll_hat = obj(p̂)

        # Convergence check: re-evaluate at a small perturbation of each parameter.
        # Threshold must be above the SLQ approximation noise (~1e-3) to avoid
        # false negatives from SLQ variation being mistaken for a better point.
        converged = true
        for i in 1:length(p̂)
            p_plus  = copy(p̂); p_plus[i]  += delta
            p_minus = copy(p̂); p_minus[i] -= delta
            if obj(p_plus) < nll_hat - 1e-2 || obj(p_minus) < nll_hat - 1e-2
                converged = false; break
            end
        end

        p̂_for_decomp = p̂

    else
        # ── 3D constrained path (rho fixed) ──────────────────────────────────────
        # rho_fixed is held constant; optimize over (sigma_a, sigma_z, sigma_eps).
        # The encoded unconstrained value for rho is atanh(rho_fixed / 0.99).
        # When rho_fixed = 0 this is 0.0, making Q block-diagonal (AKM model).
        p1_fixed = atanh(rho_fixed / 0.99)
        @printf("FixedRho: rho=%.6f (encoded p1=%.6g)\n", rho_fixed, p1_fixed)

        # 3D/4D objective: splice p1_fixed into the full parameter vector.
        function obj_fixed(p3)
            eval_counter[] += 1
            if verbose && (eval_counter[] == 1 || eval_counter[] % 10 == 0)
                if length(p3) == 4
                    @printf("[obj_fixed] eval=%d | p=(%.3g, %.3g, %.3g, %.3g) [rho=%.4f fixed]\n",
                            eval_counter[], p3[1], p3[2], p3[3], p3[4], rho_fixed)
                else
                    @printf("[obj_fixed] eval=%d | p=(%.3g, %.3g, %.3g) [rho=%.4f fixed]\n",
                            eval_counter[], p3[1], p3[2], p3[3], rho_fixed)
                end
            end
            p_full = prep.rho_eps_estimate ?
                [p1_fixed, p3[1], p3[2], p3[3], p3[4]] :
                [p1_fixed, p3[1], p3[2], p3[3]]
            st = stats_for_objective!(p_full)
            return nll_hutch_cached(
                p_full,
                st.ydot, st.projected_y, prep.K, cache;
                m=HUTCH_M, k=HUTCH_K, seed=seed, jitter=1e-12,
                cg_tol=CG_TOL, cg_maxiter=CG_MAXITER,
                log_weight_sum=st.log_weight_sum,
                within_ss=prep.within_ss,
                within_df=prep.within_df,
                rho_eps=st.rho_eps
            )
        end

        p0_3d = [log(0.7), log(0.04), log(0.4)]
        if prep.rho_eps_estimate
            push!(p0_3d, rhoeps_to_unconstrained(0.5))
        end
        if length(p0_3d) == 4
            @printf("Initial p0_3d = (%.6g, %.6g, %.6g, %.6g)\n",
                    p0_3d[1], p0_3d[2], p0_3d[3], p0_3d[4])
        else
            @printf("Initial p0_3d = (%.6g, %.6g, %.6g)\n", p0_3d[1], p0_3d[2], p0_3d[3])
        end

        t_opt = @elapsed (res = optimize(obj_fixed, p0_3d, NelderMead(), nm_opts))
        p̂3 = Optim.minimizer(res)

        ρ̂  = rho_fixed
        σâ = exp(p̂3[1]) * prep.y_std
        σẑ = exp(p̂3[2]) * prep.y_std
        σê = exp(p̂3[3]) * prep.y_std
        rho_eps_hat = prep.rho_eps_estimate ? rhoeps_from_unconstrained(p̂3[4]) : prep.rho_eps_likelihood
        nll_hat = obj_fixed(p̂3)

        converged = true
        for i in 1:length(p̂3)
            p_plus  = copy(p̂3); p_plus[i]  += delta
            p_minus = copy(p̂3); p_minus[i] -= delta
            if obj_fixed(p_plus) < nll_hat - 1e-2 || obj_fixed(p_minus) < nll_hat - 1e-2
                converged = false; break
            end
        end

        p̂_for_decomp = prep.rho_eps_estimate ?
            [p1_fixed, p̂3[1], p̂3[2], p̂3[3], p̂3[4]] :
            [p1_fixed, p̂3[1], p̂3[2], p̂3[3]]
    end

    @printf("\n--- Done ---\n")
    @printf("Nelder-Mead: %d iters, %.1fs\n", Optim.iterations(res), t_opt)
    @printf("Converged: %s | obj evals: %d\n",
            string(converged), eval_counter[])
    @printf("NLL (scaled): %.6f\n", nll_hat)
    @printf("Estimates: rho=%.6f, sigma_a=%.6f, sigma_z=%.6f, sigma_eps=%.6f\n",
            ρ̂, σâ, σẑ, σê)
    if prep.obs_weighting == :effective
        @printf("Residual correlation: rho_eps=%.6f (%s)\n",
                rho_eps_hat, prep.rho_eps_estimate ? "estimated" : "fixed")
    end

    # Optional variance decompositions (prior + posterior).
    # Fresh QOp/MOp are constructed inside each function to avoid using
    # cache.qop, which is left in a perturbed state by the convergence checks.
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

    decomp      = nothing
    post_decomp = nothing
    if decomp_probes > 0
        decomp = hutch_variance_decomp(prep_final, p̂_for_decomp;
            target=decomp_target, m=decomp_probes, seed=seed, cg_tol=CG_TOL, cg_maxiter=CG_MAXITER,
            verbose=verbose)
        post_decomp = hutch_posterior_decomp(prep_final, p̂_for_decomp;
            target=decomp_target, m=decomp_probes, seed=seed, cg_tol=CG_TOL, cg_maxiter=CG_MAXITER,
            verbose=verbose)
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
            converged=converged,
            iterations=Optim.iterations(res),
            obj_evals=eval_counter[], opt_time=t_opt,
            nll=nll_hat, rho=ρ̂, sigma_a=σâ, sigma_z=σẑ, sigma_eps=σê,
            rho_fixed=rho_fixed,
            decomp=decomp, post_decomp=post_decomp)
end

# ============================================================
# 8) Path utilities
# ============================================================

"""
Parse chunk and sample from temp/samples path.
Expected format: temp/samples/<chunk>/<sample>/edgelist.parquet
Returns: (chunk, sample, out_path)
"""
function parse_sample_path(in_path::String)::Tuple{String,String,String}
    # Normalize path separators
    normalized = replace(in_path, "\\" => "/")
    parts = split(normalized, '/')

    # Find "temp" and "samples" in path
    temp_idx = findfirst(==("temp"), parts)
    samples_idx = findfirst(==("samples"), parts)

    if temp_idx === nothing || samples_idx === nothing || samples_idx != temp_idx + 1
        error("Path must contain 'temp/samples' directory structure: $in_path")
    end

    # Extract chunk and all sample path segments (everything between samples/ and the filename)
    if length(parts) < samples_idx + 2
        error("Path must have format temp/samples/<chunk>/<sample>/file.parquet: $in_path")
    end

    chunk = parts[samples_idx + 1]
    # Collect all segments between chunk and the filename as the sample id
    sample_parts = parts[samples_idx + 2 : end - 1]
    sample = join(sample_parts, "/")

    # Construct output path: output/gmrfmle/<chunk>/<sample...>/estimates.txt
    out_dir = joinpath("output", "gmrfmle", chunk, sample_parts...)
    out_path = joinpath(out_dir, "estimates.txt")

    return (chunk, sample, out_path)
end

# ============================================================
# 9) Main
# ============================================================

function main()
    # Separate positional args from flags (--flag or --flag=value)
    positional = filter(a -> !startswith(a, "--"), ARGS)
    flags      = filter(a ->  startswith(a, "--"), ARGS)

    if length(positional) < 1
        println(stderr,
            "Usage:\n" *
            "  julia --project=. src/estimate/gmrfmle.jl <path/to/edgelist.parquet> [outcome] [flags]\n" *
            "\n" *
            "Flags:\n" *
            "  --decompose        Run variance decomposition with 200 Hutchinson probes (default)\n" *
            "  --decompose=N      Run variance decomposition with N probes\n" *
            "  --no-decompose     Skip variance decomposition\n" *
            "  --fix-rho=<v>      Fix rho to value v (abs(v) < 0.99)\n" *
            "  --maxdeg=N         Remove firms/managers with degree > N before estimation\n" *
            "  --a_weighting=<m>  Adjacency weighting: degree (default), spectral, unweighted\n" *
            "  --prior_adjacency=<m>  Prior graph adjacency: binary (default), counts\n" *
            "  --obs_weighting=<m>  Observation model: raw (default), edge, effective\n" *
            "  --rho_eps=<v|estimate>  Within-match residual correlation for effective weighting\n" *
            "  --decomp_target=<m>  Variance target: estimation (default), personyear, edge\n" *
            "\n" *
            "Input path format:\n" *
            "  temp/samples/<chunk>/<sample>/edgelist.parquet\n" *
            "\n" *
            "Output:\n" *
            "  output/gmrfmle/<chunk>/<sample>/estimates.txt  (default)\n" *
            "  Flags redirect output to parallel directories to avoid overwriting.\n" *
            "\n" *
            "Examples:\n" *
            "  julia src/estimate/gmrfmle.jl temp/samples/full/full/edgelist.parquet\n" *
            "  julia src/estimate/gmrfmle.jl temp/samples/full/full/edgelist.parquet lnR --decompose=200\n" *
            "  julia src/estimate/gmrfmle.jl temp/samples/full/full/edgelist.parquet lnR --fix-rho=0\n" *
            "  julia src/estimate/gmrfmle.jl temp/samples/full/full/edgelist.parquet lnR --maxdeg=20 --a_weighting=unweighted\n" *
            "  julia src/estimate/gmrfmle.jl temp/samples/personyear/full/giant/edgelist.parquet lnR --prior_adjacency=counts\n" *
            "  julia src/estimate/gmrfmle.jl temp/samples/personyear/full/minedge-5/edgelist.parquet lnR --obs_weighting=effective --rho_eps=estimate\n"
        )
        return 1
    end

    in_path = positional[1]
    outcome = (length(positional) >= 2) ? Symbol(positional[2]) : :lnR

    # Parse flags; --decompose=200 is the default
    decomp_probes = 200
    rho_fixed     = nothing
    maxdeg        = nothing
    a_weighting   = :degree
    prior_adjacency = :binary
    obs_weighting = :raw
    rho_eps_fixed = nothing
    rho_eps_estimate = false
    decomp_target = :estimation
    for flag in flags
        if flag == "--decompose"
            decomp_probes = 200
        elseif startswith(flag, "--decompose=")
            decomp_probes = parse(Int, split(flag, '=', limit=2)[2])
        elseif flag == "--no-decompose"
            decomp_probes = 0
        elseif startswith(flag, "--fix-rho=")
            rho_fixed = parse(Float64, split(flag, '=', limit=2)[2])
            abs(rho_fixed) >= 0.99 && error("--fix-rho value must satisfy abs(rho) < 0.99, got $rho_fixed")
        elseif startswith(flag, "--maxdeg=")
            maxdeg = parse(Int, split(flag, '=', limit=2)[2])
            maxdeg >= 1 || error("--maxdeg must be >= 1, got $maxdeg")
        elseif startswith(flag, "--a_weighting=")
            a_weighting = Symbol(split(flag, '=', limit=2)[2])
            a_weighting in (:degree, :spectral, :unweighted) ||
                error("--a_weighting must be degree, spectral, or unweighted; got $a_weighting")
        elseif startswith(flag, "--prior_adjacency=")
            prior_adjacency = Symbol(split(flag, '=', limit=2)[2])
            prior_adjacency in (:binary, :counts) ||
                error("--prior_adjacency must be binary or counts; got $prior_adjacency")
        elseif startswith(flag, "--obs_weighting=")
            obs_weighting = Symbol(split(flag, '=', limit=2)[2])
            obs_weighting in (:raw, :edge, :effective) ||
                error("--obs_weighting must be raw, edge, or effective; got $obs_weighting")
        elseif startswith(flag, "--rho_eps=")
            val = split(flag, '=', limit=2)[2]
            if lowercase(strip(val)) == "estimate"
                rho_eps_estimate = true
                rho_eps_fixed = nothing
            else
                rho_eps_fixed = parse(Float64, val)
                0.0 <= rho_eps_fixed < 1.0 ||
                    error("--rho_eps must satisfy 0 <= rho_eps < 1, got $rho_eps_fixed")
                rho_eps_estimate = false
            end
        elseif startswith(flag, "--decomp_target=")
            decomp_target = normalize_decomp_target(Symbol(split(flag, '=', limit=2)[2]))
        end
    end
    if obs_weighting == :effective && !rho_eps_estimate && rho_eps_fixed === nothing
        rho_eps_estimate = true
    end
    if obs_weighting != :effective && (rho_eps_estimate || rho_eps_fixed !== nothing)
        error("--rho_eps can only be used with --obs_weighting=effective")
    end

    isfile(in_path) || error("Input file not found: $in_path")

    # Parse paths
    chunk, sample_id, out_path = parse_sample_path(in_path)

    # Redirect output to parallel directories so non-default runs never
    # overwrite the baseline estimates.  Build a suffix from active flags.
    out_suffix = ""
    if rho_fixed !== nothing
        rho_str = @sprintf("%g", rho_fixed)
        out_suffix *= "-rho$(rho_str)"
    end
    if a_weighting != :degree
        out_suffix *= "-$(a_weighting)"
    end
    if prior_adjacency != :binary
        out_suffix *= "-prior-$(prior_adjacency)"
    end
    if maxdeg !== nothing
        out_suffix *= "-maxdeg$(maxdeg)"
    end
    if obs_weighting != :raw
        out_suffix *= "-obs-$(obs_weighting)"
    end
    if obs_weighting == :effective
        if rho_eps_estimate
            out_suffix *= "-rhoeps-est"
        elseif rho_eps_fixed !== nothing
            out_suffix *= "-rhoeps$(safe_float_id(rho_eps_fixed))"
        end
    end
    if decomp_target != :estimation
        out_suffix *= "-decomp-$(decomp_target)"
    end
    if out_suffix != ""
        out_path = replace(out_path, "output/gmrfmle/" => "output/gmrfmle$(out_suffix)/", count=1)
    end

    out_dir = dirname(out_path)
    mkpath(out_dir)

    @printf("Reading: %s\n", in_path)
    df = Parquet2.readfile(in_path) |> DataFrame
    @printf("Loaded rows: %d\n", nrow(df))

    if maxdeg !== nothing
        df = filter_maxdeg(df, maxdeg; verbose=true)
        nrow(df) > 0 || error("No edges remain after maxdeg=$maxdeg filter.")
    end

    # Unweighted Q has ~d_max/d_min worse conditioning than degree-weighted,
    # so SLQ needs more Lanczos steps (HUTCH_K) to accurately approximate
    # log-det across the wider spectral range at high rho.
    hutch_m = a_weighting == :unweighted ? 16 : 30
    hutch_k = a_weighting == :unweighted ? 50 : 30

    r = estimate_hutch(
        df;
        outcome=outcome,
        a_weighting=a_weighting,
        prior_adjacency=prior_adjacency,
        obs_weighting=obs_weighting,
        rho_eps_fixed=rho_eps_fixed,
        rho_eps_estimate=rho_eps_estimate,
        HUTCH_M=hutch_m,
        HUTCH_K=hutch_k,
        CG_TOL=1e-6,
        CG_MAXITER=700,
        iters=1000,
        seed=20260126,
        verbose=true,
        decomp_probes=decomp_probes,
        decomp_target=decomp_target,
        rho_fixed=rho_fixed
    )

    open(out_path, "w") do io
        @printf(io, "GMRF MLE Estimation (Hutch/SLQ + PCG, Nelder-Mead)\n")
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
        if maxdeg !== nothing
            @printf(io, "MaxDeg=%d\n", maxdeg)
        end
        if r.rho_fixed !== nothing
            @printf(io, "FixedRho=%.6f  (constrained run; 3D optimization over sigma_a, sigma_z, sigma_eps)\n", r.rho_fixed)
        end
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
            @printf(io, "\nPrior Variance Decomposition (Object 2; target=%s, Hutchinson, m=%d, PCG converged %d/%d):\n",
                    String(d.target), d.probes, d.pcg_converged, d.probes)
            @printf(io, "  TargetWeightSum = %.6f | ResidualLevel=%s | ObservedSecondMoment=%.6f\n",
                    d.weight_sum, String(d.residual_level), d.observed_second_moment)
            @printf(io, "  V_firm     = %+.6f  (target-weighted firm effects)\n",  d.V_firm)
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
            @printf(io, "\nPosterior Variance Decomposition (Object 3; target=%s, Hutchinson, m=%d, PCG traces %d/%d):\n",
                    String(p.target), p.probes, p.pcg_converged_traces, p.probes)
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
