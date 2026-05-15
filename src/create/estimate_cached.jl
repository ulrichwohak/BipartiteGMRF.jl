#!/usr/bin/env julia
# estimate_2010_lnR_hutch_cached.jl
#
# Standalone estimator (mirrors estimate.jl; no includes).
# Default input: temp/edgelist.parquet
# Fixed target: years=2016-2018 (full sample), outcome=lnR
# No weights.

using Parquet2
using DataFrames
using Optim
using FiniteDiff
using SparseArrays
using LinearAlgebra
using Statistics
using Random
using Printf: @sprintf, @printf

# ============================================================
# 1) Utilities
# ============================================================

speye(n::Int) = spdiagm(0 => ones(Float64, n))

# Robust year parsing / coercion
year_to_int(x) = x isa Missing ? missing :
    (x isa Integer ? Int(x) :
     x isa AbstractFloat ? Int(round(x)) :
     x isa AbstractString ? parse(Int, x) :
     Int(x))

# Robust float coercion with missing -> NaN
to_float_nan(x) = x isa Missing ? NaN : Float64(x)

# ============================================================
# 2) Prior precision Q construction (SPD-by-construction via normalized W)
# ============================================================

function build_precision_bipartite(A_fm::SparseMatrixCSC{Float64,Int}, rho, sigma_a, sigma_z)
    N_f, N_m = size(A_fm)

    d_f = vec(sum(A_fm; dims=2))
    d_m = vec(sum(A_fm; dims=1))

    if any(d_f .<= 0) || any(d_m .<= 0)
        error("A_fm has zero-degree nodes; cannot normalize. Ensure every node degree ≥ 1.")
    end

    Df_inv_sqrt = spdiagm(0 => 1.0 ./ sqrt.(d_f))
    Dm_inv_sqrt = spdiagm(0 => 1.0 ./ sqrt.(d_m))

    W = Df_inv_sqrt * A_fm * Dm_inv_sqrt

    Q_ff = (1.0 / sigma_a^2) * speye(N_f)
    Q_mm = (1.0 / sigma_z^2) * speye(N_m)
    Q_fm = -(rho / (sigma_a * sigma_z)) * W

    Q = [Q_ff Q_fm;
         Q_fm' Q_mm]
    return Q
end

# ============================================================
# 3) Correct sparse sufficient statistics for V:
#    projected_y = V'y
#    VtV = V'V = [diag(c_f) A_obs; A_obs' diag(c_m)]
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

    # Pair co-occurrence counts (off-diagonal block) == multiplicities in the year
    A_obs = sparse(f_rows, m_cols, ones(Float64, K), N_f, N_m)

    VtV = [spdiagm(0 => cnt_f)  A_obs;
           A_obs'              spdiagm(0 => cnt_m)]

    return projected_y, VtV
end

# ============================================================
# 4) Preconditioned Conjugate Gradient (Jacobi)
# ============================================================

function pcg_solve(
    mulA!::Function,
    b::AbstractVector{<:Real};
    tol::Real=1e-6,
    maxiter::Int=700,
    Mdiag::Union{Nothing,AbstractVector{<:Real}}=nothing,
    diag_floor::Real=1e-12
)
    n = length(b)
    x  = zeros(Float64, n)
    r  = Vector{Float64}(undef, n)
    z  = Vector{Float64}(undef, n)
    p  = Vector{Float64}(undef, n)
    Ap = Vector{Float64}(undef, n)

    @inbounds for i in 1:n
        r[i] = Float64(b[i])  # x=0 => r=b
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
# 5) SLQ logdet for SPD via matvec only
# ============================================================

function slq_logdet_spd_mul!(mulA!::Function, n::Int;
    m::Int=30, k::Int=30,
    seed::Int=1,
    jitter::Real=1e-12,
    ritz_floor::Real=1e-14
)
    k_eff = min(k, n)
    rng = MersenneTwister(seed)

    z  = Vector{Float64}(undef, n)
    q  = zeros(Float64, n)
    q0 = zeros(Float64, n)
    w  = zeros(Float64, n)

    total = 0.0
    nz = sqrt(n)

    for _ in 1:m
        @inbounds for i in 1:n
            z[i] = rand(rng, Bool) ? 1.0 : -1.0
        end

        @. q = z / nz
        fill!(q0, 0.0)
        beta_prev = 0.0

        α = zeros(Float64, k_eff)
        β = zeros(Float64, max(k_eff-1, 0))

        for t in 1:k_eff
            mulA!(w, q)
            if jitter != 0
                @. w = w + jitter*q
            end

            α[t] = dot(q, w)
            @. w = w - α[t]*q - beta_prev*q0

            if t < k_eff
                beta_prev = norm(w)
                if !isfinite(beta_prev)
                    return NaN
                end
                β[t] = beta_prev

                if beta_prev < 1e-14
                    α = α[1:t]
                    β = β[1:max(t-1, 0)]
                    break
                end

                q0 .= q
                @. q = w / beta_prev
            end
        end

        if any(!isfinite, α) || any(!isfinite, β)
            return NaN
        end

        T = SymTridiagonal(α, β)
        local E
        try
            E = eigen(T)
        catch
            return NaN
        end
        λ = max.(E.values, ritz_floor)
        u1 = E.vectors[1, :]

        total += (nz^2) * sum(log.(λ) .* (u1 .^ 2))
    end

    return total / m
end

function slq_logdet_spd(A::AbstractMatrix; m::Int=30, k::Int=30,
    seed::Int=1, jitter::Real=1e-12, ritz_floor::Real=1e-14
)
    n = size(A,1)
    As = Symmetric(A)
    mulA!(y, x) = mul!(y, As, x)
    return slq_logdet_spd_mul!(mulA!, n; m=m, k=k, seed=seed, jitter=jitter, ritz_floor=ritz_floor)
end

# ============================================================
# 6) Hutchinson objective: CG quad + SLQ logdets
# ============================================================

function nll_hutch(
    params,
    ydot,
    projected_y,
    VtV,
    A_fm,
    K;
    m::Int=30, k::Int=30, seed::Int=20260126,
    jitter::Real=1e-12, ritz_floor::Real=1e-14,
    cg_tol::Real=1e-6,
    cg_maxiter::Int=700,
    cg_diag_floor::Real=1e-12
)
    BIG = 1e30

    ρ  = 0.99 * tanh(params[1])
    σa = exp(params[2])
    σz = exp(params[3])
    σe = exp(params[4])

    if !(isfinite(ρ) && isfinite(σa) && isfinite(σz) && isfinite(σe)) || σe <= 0 || σa <= 0 || σz <= 0
        return BIG
    end

    λ = 1.0 / σe^2

    Q = build_precision_bipartite(A_fm, ρ, σa, σz)
    Qs = Symmetric(Q)

    n = length(projected_y)

    # Matvec for M = Q + λ*VtV without forming M explicitly
    tmpMV = zeros(Float64, n)
    function mulM!(y, x)
        mul!(y, Qs, x)
        mul!(tmpMV, VtV, x)
        @. y = y + λ*tmpMV
        return y
    end

    # Jacobi preconditioner diag(M) = diag(Q) + λ*diag(VtV)
    dV = diag(VtV)
    Mdiag = similar(dV)

    inv_sa2 = 1.0 / (σa^2)
    inv_sz2 = 1.0 / (σz^2)
    N_f = size(A_fm, 1)

    @inbounds for i in 1:n
        dq = (i <= N_f) ? inv_sa2 : inv_sz2
        Mdiag[i] = dq + λ * dV[i]
    end

    # Quad term via (P)CG
    x, ok, iters, relres = pcg_solve(mulM!, projected_y;
        tol=cg_tol, maxiter=cg_maxiter, Mdiag=Mdiag, diag_floor=cg_diag_floor
    )
    if !ok
        return BIG
    end

    quad_term = dot(projected_y, x)
    if !isfinite(quad_term)
        return BIG
    end

    # SLQ logdets
    ldQ = slq_logdet_spd(Q; m=m, k=k, seed=seed + 0, jitter=jitter, ritz_floor=ritz_floor)
    ldM = slq_logdet_spd_mul!(mulM!, n; m=m, k=k, seed=seed + 10_000, jitter=jitter, ritz_floor=ritz_floor)

    if !(isfinite(ldQ) && isfinite(ldM))
        return BIG
    end

    val = 0.5 * ( (K * (2.0*log(σe))) + (ldM - ldQ) + (λ * ydot) - ((λ^2) * quad_term) )
    return isfinite(val) ? val : BIG
end

# ============================================================
# 7) Data mapping: Parquet -> DataFrame -> model inputs
# ============================================================

function prepare_year_window(
    df::DataFrame;
    year_start::Int,
    year_end::Int,
    outcome::Symbol,
    sample_frac::Float64=1.0,
    seed::Int=20260126
)
    required = [:frame_id_numeric, :person_id, :year, outcome]
    for c in required
        hasproperty(df, c) || error("Missing required column: $(c)")
    end

    years = map(year_to_int, df.year)
    mask_year = map(y -> (y !== missing) && (year_start <= y <= year_end), years)

    d = df[mask_year, :]
    if nrow(d) == 0
        error("No rows with year in $(year_start)-$(year_end).")
    end

    y_raw_all = map(to_float_nan, d[!, outcome])
    keep = map(isfinite, y_raw_all)
    d = d[keep, :]
    y_raw = Float64.(y_raw_all[keep])

    if !(0.0 < sample_frac <= 1.0)
        error("sample_frac must be in (0, 1]. Got $(sample_frac).")
    end
    if sample_frac < 1.0
        rng = MersenneTwister(seed)
        sample_mask = rand(rng, nrow(d)) .< sample_frac
        d = d[sample_mask, :]
        y_raw = y_raw[sample_mask]
    end

    if length(y_raw) == 0
        error("No usable observations after filtering to years=$(year_start)-$(year_end) and non-missing $(outcome).")
    end

    # Build contiguous indices
    firms  = unique(d.frame_id_numeric)
    people = unique(d.person_id)

    N_F = length(firms)
    N_M = length(people)
    K   = nrow(d)

    firm_to_idx   = Dict{eltype(firms),Int}(f => i for (i,f) in enumerate(firms))
    person_to_idx = Dict{eltype(people),Int}(p => i for (i,p) in enumerate(people))

    f_rows = Vector{Int}(undef, K)
    m_cols = Vector{Int}(undef, K)

    @inbounds for k in 1:K
        f_rows[k] = firm_to_idx[d.frame_id_numeric[k]]
        m_cols[k] = person_to_idx[d.person_id[k]]
    end

    # Center within window, then scale
    μy = mean(y_raw)
    y_raw .-= μy

    y_std = std(y_raw)
    if !(isfinite(y_std) && y_std > 0)
        error("Outcome has zero/invalid std in years=$(year_start)-$(year_end). Cannot scale.")
    end
    y = y_raw ./ y_std
    ydot = dot(y, y)

    # V statistics (V'y and V'V with correct off-diagonals via A_obs)
    projected_y, VtV = build_V_stats(f_rows, m_cols, y, N_F, N_M)

    # Prior adjacency A_fm should be BINARY connectivity, not counts.
    # Build from the same (f_rows, m_cols) edges and then squash nzval to 1.
    A_fm = sparse(f_rows, m_cols, ones(Float64, K), N_F, N_M)
    A_fm.nzval .= 1.0

    # Defensive degree check
    d_f = vec(sum(A_fm; dims=2))
    d_m = vec(sum(A_fm; dims=1))
    if any(d_f .<= 0) || any(d_m .<= 0)
        error("Zero-degree node detected after indexing from filtered sample; investigate year window / IDs.")
    end

    return (y=y, ydot=ydot, y_std=y_std,
            f_rows=f_rows, m_cols=m_cols,
            A_fm=A_fm, projected_y=projected_y, VtV=VtV,
            K=K, N_F=N_F, N_M=N_M)
end

# ============================================================
# 8) Estimation wrapper for year=2010, lnR
# ============================================================

function estimate_2016_2018_lnR_hutch(
    df::DataFrame;
    year_start::Int=2016,
    year_end::Int=2018,
    outcome::Symbol=:lnR,
    sample_frac::Float64=1.0,
    output_path::AbstractString="temp/gmrf_results.txt",
    # SLQ knobs
    HUTCH_M::Int=30,
    HUTCH_K::Int=30,
    # CG knobs
    CG_TOL::Float64=1e-6,
    CG_MAXITER::Int=700,
    # optimizer knobs
    iters::Int=250,
    seed::Int=20260126
)
    prep = prepare_year_window(
        df;
        year_start=year_start,
        year_end=year_end,
        outcome=outcome,
        sample_frac=sample_frac,
        seed=seed
    )

    n = prep.N_F + prep.N_M
    @printf("Years=%d-%d, outcome=%s\n", year_start, year_end, String(outcome))
    @printf("Sample frac=%.2f\n", sample_frac)
    @printf("N_F=%d, N_M=%d, K=%d, n=%d\n", prep.N_F, prep.N_M, prep.K, n)
    @printf("y_std=%.6g (scaling factor)\n", prep.y_std)

    # objective
    obj(p) = nll_hutch(
        p, prep.ydot, prep.projected_y, prep.VtV, prep.A_fm, prep.K;
        m=HUTCH_M, k=HUTCH_K, seed=seed, jitter=1e-12,
        cg_tol=CG_TOL, cg_maxiter=CG_MAXITER
    )

    # finite-diff gradient (no GradientCache pitfalls)
    function grad!(G, p)
        FiniteDiff.finite_difference_gradient!(G, obj, p)
    end

    # init in transformed params
    p0 = [atanh(0.5), 0.0, 0.0, 0.0]

    od = OnceDifferentiable(obj, grad!, p0)
    res = optimize(od, p0, LBFGS(), Optim.Options(iterations=iters, show_trace=true))

    p̂ = Optim.minimizer(res)

    # transform back and rescale sigmas
    ρ̂  = 0.99 * tanh(p̂[1])
    σâ = exp(p̂[2]) * prep.y_std
    σẑ = exp(p̂[3]) * prep.y_std
    σê = exp(p̂[4]) * prep.y_std

    @printf("\nConverged: %s\n", string(Optim.converged(res)))
    @printf("Iterations: %d\n", Optim.iterations(res))
    @printf("NLL (scaled): %.6f\n", obj(p̂))
    @printf("\nEstimates (structural units):\n")
    @printf("rho        = %.6f\n", ρ̂)
    @printf("sigma_a    = %.6f\n", σâ)
    @printf("sigma_z    = %.6f\n", σẑ)
    @printf("sigma_eps  = %.6f\n", σê)

    open(output_path, "w") do io
        @printf(io, "Years=%d-%d, outcome=%s\n", year_start, year_end, String(outcome))
        @printf(io, "Sample frac=%.2f\n", sample_frac)
        @printf(io, "N_F=%d, N_M=%d, K=%d, n=%d\n", prep.N_F, prep.N_M, prep.K, n)
        @printf(io, "y_std=%.6g (scaling factor)\n", prep.y_std)
        @printf(io, "\nConverged: %s\n", string(Optim.converged(res)))
        @printf(io, "Iterations: %d\n", Optim.iterations(res))
        @printf(io, "NLL (scaled): %.6f\n", obj(p̂))
        @printf(io, "\nEstimates (structural units):\n")
        @printf(io, "rho        = %.6f\n", ρ̂)
        @printf(io, "sigma_a    = %.6f\n", σâ)
        @printf(io, "sigma_z    = %.6f\n", σẑ)
        @printf(io, "sigma_eps  = %.6f\n", σê)
    end
    @printf("Results written to %s\n", output_path)

    return (res=res, p_hat=p̂, rho=ρ̂, sigma_a=σâ, sigma_z=σẑ, sigma_eps=σê)
end

# ============================================================
# 9) Main
# ============================================================

function main()
    path = (length(ARGS) >= 1) ? ARGS[1] : "temp/edgelist.parquet"
    @printf("Reading parquet: %s\n", path)

    # Parquet -> DataFrame
    df = Parquet2.readfile(path) |> DataFrame

    # Fixed as requested
    estimate_2016_2018_lnR_hutch(
        df;
        year_start=2016,
        year_end=2018,
        outcome=:lnR,
        sample_frac=1.0,
        output_path="temp/gmrf_results.txt"
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
