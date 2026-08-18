# ═══════════════════════════════════════════════════════════════════════════
# Free per-firm error blocks: EM estimator (issue #112)
#
# Data stay at edge level. Each firm's rows carry a free PD error covariance
# Ω_i estimated by EM with a closed-form M-step; the GMRF hyperparameters
# (ρ, σ_a, σ_z) take one damped Newton step per EM iteration.
# ═══════════════════════════════════════════════════════════════════════════

# Firm and worker node column of each observation row, read once from the
# edge-level design `V` (row t has exactly two nonzeros: one firm column ≤ nf,
# one worker column > nf).
function _edge_nodes(V::SparseMatrixCSC{Float64,Int}, nf::Int)
    K = size(V, 1)
    f = zeros(Int, K)
    w = zeros(Int, K)
    rows, cols, _ = findnz(V)
    for (r, c) in zip(rows, cols)
        if c <= nf
            f[r] = c
        else
            w[r] = c - nf
        end
    end
    return f, w
end

# Observation gaps: gap i lists the edge-level row indices whose block_of is i.
function _block_rows(fb::FreeBlockStats)
    B = length(fb.sizes)
    rows = [Int[] for _ in 1:B]
    for s in eachindex(fb.block_of)
        push!(rows[fb.block_of[s]], s)
    end
    return rows
end

"""
    assemble_freeblock_precision(fb, Ωs, nf, nw) -> (P_data, b)

Assemble the data term `A'Ω⁻¹A` (n×n sparse) and `A'Ω⁻¹y` (n) from the per-firm
blocks `Ωs` (a `Vector{Matrix{Float64}}`, one PD block per firm). `A = [D F]`
is the edge-level design; Ω is block-diagonal by firm, so the only structural
fill-in over the iid case is the manager-manager clique per firm. No global Ω
is materialized. Throws if a block Cholesky fails (non-PD guardrail).
"""
function assemble_freeblock_precision(fb::FreeBlockStats, Ωs::Vector{Matrix{Float64}}, nf::Int, nw::Int)
    n = nf + nw
    f, w = _edge_nodes(fb.V, nf)
    rows = _block_rows(fb)
    length(Ωs) == length(rows) ||
        throw(ArgumentError("one Ω per firm block expected."))

    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, sum(m -> m * m + 2m + 1, fb.sizes))
    sizehint!(J, sum(m -> m * m + 2m + 1, fb.sizes))
    sizehint!(V, sum(m -> m * m + 2m + 1, fb.sizes))
    b = zeros(Float64, n)

    for i in eachindex(rows)
        r = rows[i]
        ms = fb.sizes[i]
        C = cholesky(Symmetric(Ωs[i]); check = false)
        issuccess(C) || throw(ArgumentError(
            "error block $i (firm $(f[first(r)])) is not positive definite (m=$ms)."))
        G = inv(C)                       # Ω_i⁻¹ (dense, block-local ordering)
        firm = f[first(r)]
        yi = fb.y[r]
        Gy = G * yi

        # D'Ω⁻¹D firm diagonal
        push!(I, firm); push!(J, firm); push!(V, sum(G))
        # D'Ω⁻¹y
        b[firm] += sum(Gy)
        for a in 1:ms, c in 1:ms
            t = r[a]; tc = r[c]
            m = nf + w[t]; mc = nf + w[tc]
            # F'Ω⁻¹F manager-manager (block-local G entries)
            push!(I, m); push!(J, mc); push!(V, G[a, c])
        end
        # D'Ω⁻¹F cross (and its transpose), and F'Ω⁻¹y
        colsum = vec(sum(G; dims = 1))   # 1'Ω_i⁻¹ = column sums
        for a in 1:ms
            t = r[a]
            m = nf + w[t]
            push!(I, firm); push!(J, m); push!(V, colsum[a])
            push!(I, m); push!(J, firm); push!(V, colsum[a])
            b[m] += Gy[a]
        end
    end
    return sparse(I, J, V, n, n), b
end

# ─── EM scaffolding ────────────────────────────────────────────────────────

"""
Workspace for the [`EMFreeBlocks`](@ref) solver. `ws_Q` factors the prior
precision `K` (pattern fixed across ρ ≠ 0); `ws_M` factors the posterior
`M = K + A'Ω⁻¹A` (pattern fixed as the full free-Ω pattern — see R1 in the
plan). `f` holds the firm node of each edge-level row (for the PD guardrail).
"""
struct EMFreeWorkspace
    ws_Q::GaussianMarkovRandomFields.GMRFWorkspace
    ws_M::GaussianMarkovRandomFields.GMRFWorkspace
    f::Vector{Int}
end

function _init_blocks(fb::FreeBlockStats)
    return [Matrix{Float64}(I, m, m) .+ 0.5 .* (ones(m, m) .- Matrix{Float64}(I, m, m))
            for m in fb.sizes]
end

# Project a symmetric block onto {λ_min ≥ λ_lo} by clamping its eigenvalues up
# to the fixed floor λ_lo (eigenvectors preserved). λ_lo is a GLOBAL,
# iteration-fixed scale — not the block's own trace — so a single pass is exact
# and the post-clamp bound holds. `eig_floor` encodes the "well-conditioned Ωᵢ"
# assumption (issue #112 §5.4): pattern-free, forbids only near-singularity.
function _floor_spectrum(S::Matrix{Float64}, λ_lo::Float64)
    F = eigen(Symmetric(S))
    F.values[1] >= λ_lo && return S
    λc = max.(F.values, λ_lo)
    return Matrix(Symmetric((F.vectors .* transpose(λc)) * F.vectors'))
end

function make_emfree_workspace(model::AbstractBipartiteModel, fb::FreeBlockStats, nf::Int, nw::Int)
    # Seed ω_M with the FULL free-Ω pattern (reference blocks carry full
    # within-firm off-diagonal support), so the symbolic factorization covers
    # the manager-manager fill-in that a diagonal (iid) seed would drop.
    P0, _ = assemble_freeblock_precision(fb, _init_blocks(fb), nf, nw)
    rho_ref = min(0.1, 0.5 * rho_limit(model)) + eps(Float64)
    Q0 = model_precision(model, rho_ref, 1.0, 1.0)
    ws_Q = GaussianMarkovRandomFields.GMRFWorkspace(Q0)
    ws_M = GaussianMarkovRandomFields.GMRFWorkspace(Q0 + P0)
    f, _ = _edge_nodes(fb.V, nf)
    return EMFreeWorkspace(ws_Q, ws_M, f)
end

# Posterior correction Aᵢ M⁻¹ Aᵢ' for the block's edge-level design rows Vᵢ,
# computed as Vᵢ · (M⁻¹ Vᵢ') by triangular solves. This Gram form avoids the
# cancellation of the 4-term selected-inverse sum when M is ill-conditioned
# (near-singular Ω blocks): each entry is an inner product of *solved* vectors,
# so the result stays PSD to roundoff.
function _aptpa(ws::GaussianMarkovRandomFields.GMRFWorkspace, Vi::SparseMatrixCSC{Float64,Int})
    m = size(Vi, 1)
    n = size(Vi, 2)
    X = Matrix{Float64}(undef, n, m)
    for a in 1:m
        X[:, a] = GaussianMarkovRandomFields.workspace_solve(ws, Vector(Vi[a, :]))
    end
    return Matrix(Vi * X)
end

"""
    emfree_e_step(model, fb, ew, Ωs, rho, sa, sz, nf, nw) -> (α̂, S_ε, b)

One E-step: factor `M = K + A'Ω⁻¹A`, solve the penalized-GLS mode
`α̂ = M⁻¹ A'Ω⁻¹y`, and form the block posterior second moments
`S_{ε,i} = r_i r_i' + A_i M⁻¹ A_i'` (the next EM iterate of `Ω_i`).
"""
function emfree_e_step(
    model::AbstractBipartiteModel,
    fb::FreeBlockStats,
    ew::EMFreeWorkspace,
    Ωs::Vector{Matrix{Float64}},
    rho::Float64,
    sa::Float64,
    sz::Float64,
    nf::Int,
    nw::Int,
)
    P, b = assemble_freeblock_precision(fb, Ωs, nf, nw)
    Q = model_precision(model, rho, sa, sz)
    GaussianMarkovRandomFields.update_precision!(ew.ws_Q, Q)
    GaussianMarkovRandomFields.ensure_numeric!(ew.ws_Q)
    GaussianMarkovRandomFields.update_precision!(ew.ws_M, Q + P)
    GaussianMarkovRandomFields.ensure_numeric!(ew.ws_M)
    alpha = GaussianMarkovRandomFields.workspace_solve(ew.ws_M, b)

    V = fb.V
    r = fb.y - V * alpha
    rows = _block_rows(fb)
    Seps = Vector{Matrix{Float64}}(undef, length(rows))
    for i in eachindex(rows)
        rr = rows[i]
        ri = r[rr]
        S = ri * ri' + _aptpa(ew.ws_M, V[rr, :])
        Seps[i] = 0.5 .* (S .+ S')   # exact symmetry; cholesky reads one triangle
    end
    return alpha, Seps, b
end

"""
    emfree_nll(model, fb, ew, Ωs, rho, sa, sz, nf, nw) -> Float64

Marginal negative log-likelihood at fixed `Ω` and `(ρ, σ_a, σ_z)`, in
standardized units and with additive constants dropped (matching the iid
objective's convention).
"""
function emfree_nll(
    model::AbstractBipartiteModel,
    fb::FreeBlockStats,
    ew::EMFreeWorkspace,
    Ωs::Vector{Matrix{Float64}},
    rho::Float64,
    sa::Float64,
    sz::Float64,
    nf::Int,
    nw::Int,
)
    try
        P, b = assemble_freeblock_precision(fb, Ωs, nf, nw)
        Q = model_precision(model, rho, sa, sz)
        GaussianMarkovRandomFields.update_precision!(ew.ws_Q, Q)
        GaussianMarkovRandomFields.ensure_numeric!(ew.ws_Q)
        GaussianMarkovRandomFields.update_precision!(ew.ws_M, Q + P)
        GaussianMarkovRandomFields.ensure_numeric!(ew.ws_M)

        logdet_Ω = 0.0
        yΩy = 0.0
        rows = _block_rows(fb)
        for i in eachindex(rows)
            C = cholesky(Symmetric(Ωs[i]))
            logdet_Ω += logdet(C)
            yi = fb.y[rows[i]]
            yΩy += dot(yi, C \ yi)
        end
        ldQ = -GaussianMarkovRandomFields.logdet_cov(ew.ws_Q)
        ldM = -GaussianMarkovRandomFields.logdet_cov(ew.ws_M)
        alpha = GaussianMarkovRandomFields.workspace_solve(ew.ws_M, b)
        return 0.5 * (logdet_Ω + (ldM - ldQ) + yΩy - dot(b, alpha))
    catch e
        e isa InterruptException && rethrow()
        return BIG_NLL
    end
end

# ─── EM loop ───────────────────────────────────────────────────────────────

function _emfree_θ_from_ψ(ψ::Vector{Float64}, limit::Float64)
    ρ = limit * tanh(clamp(ψ[1], -25.0, 25.0))
    return ρ, exp(ψ[2]), exp(ψ[3])
end

function _emfree_ψ_from_θ(rho::Float64, sa::Float64, sz::Float64, limit::Float64)
    return [atanh(rho / limit), log(sa), log(sz)]
end

"""
Exact marginal score (Fisher identity) of the NLL w.r.t. `(ρ, σ_a, σ_z)`, in
ψ-space. The complete-data score `∂ℓ/∂θ_k = ½ tr(∂K/∂θ_k (K⁻¹ − S_α))` equals
the marginal score at any interior point (§5.1); with `S_α = α̂α̂' + M⁻¹` this
reads entirely off `selinv_dot` of the already-factored `K` and `M` workspaces
plus the penalized-GLS mode `α̂`. `∂K/∂θ` is finite-differenced (K is cheap to
build). Returns the NLL gradient mapped to ψ through the codec Jacobian.
"""
function _emfree_score(
    model::BipartiteVarianceStableModel,
    ew::EMFreeWorkspace,
    alpha::Vector{Float64},
    ψ::Vector{Float64},
    limit::Float64,
)
    rho, sa, sz = _emfree_θ_from_ψ(ψ, limit)
    θ = [rho, sa, sz]
    g = zeros(Float64, 3)
    for k in 1:3
        # relative step: keeps σ_a, σ_z strictly positive; ρ gets a small floor
        # so a near-zero ρ still supports a finite difference.
        h = k == 1 ? sqrt(eps(Float64)) * max(abs(θ[k]), 1e-6) :
                     sqrt(eps(Float64)) * θ[k]
        θp = copy(θ); θm = copy(θ)
        θp[k] += h; θm[k] -= h
        # clamp probes into the valid domain (ρ interior, σ's positive) so an
        # off-boundary finite difference never trips _validate_bipartite_params.
        θp[1] = clamp(θp[1], -limit + 1e-6, limit - 1e-6)
        θm[1] = clamp(θm[1], -limit + 1e-6, limit - 1e-6)
        θp[2] = max(θp[2], 1e-10); θm[2] = max(θm[2], 1e-10)
        θp[3] = max(θp[3], 1e-10); θm[3] = max(θm[3], 1e-10)
        dK = model_precision(model, θp[1], θp[2], θp[3]) -
             model_precision(model, θm[1], θm[2], θm[3])
        dK = dK * (1.0 / (2h))
        trK = GaussianMarkovRandomFields.selinv_dot(ew.ws_Q, dK)   # tr(∂K · K⁻¹)
        trM = GaussianMarkovRandomFields.selinv_dot(ew.ws_M, dK)   # tr(∂K · M⁻¹)
        quad = dot(alpha, dK * alpha)
        g[k] = 0.5 * (quad + trM - trK)                            # ∂NLL/∂θ_k
    end
    # chain rule: dρ/dψ₁ = limit·sech²(ψ₁) = limit − ρ²/limit, dσ/dlog = σ
    return [g[1] * (limit - rho^2 / limit), g[2] * sa, g[3] * sz]
end

"""
One safeguarded score step on the GMRF parameters with the blocks held fixed:
move along the negative Fisher-identity gradient with Armijo backtracking on
the marginal NLL (monotone descent; never a derivative-free nested solve).
"""
function _emfree_theta_step(
    model::BipartiteVarianceStableModel,
    fb::FreeBlockStats,
    ew::EMFreeWorkspace,
    Ωs::Vector{Matrix{Float64}},
    ψ0::Vector{Float64},
    gψ::Vector{Float64},
    limit::Float64,
    nf::Int,
    nw::Int,
)
    f = ψ -> emfree_nll(model, fb, ew, Ωs, _emfree_θ_from_ψ(ψ, limit)..., nf, nw)
    f0 = f(ψ0)
    step = 1.0
    for _ in 1:80
        ψn = ψ0 .- step .* gψ
        fn = f(ψn)
        isfinite(fn) && fn < f0 && return ψn
        step *= 0.5
    end
    return ψ0
end

"""
    optimize_emfree(model, stats, solver; verbose) -> NamedTuple

Run the [`EMFreeBlocks`](@ref) loop to convergence: alternate a closed-form
M-step on the per-firm blocks `Ω_i` with one safeguarded score step on the GMRF
hyperparameters (η pulled from the Fisher-identity score, monotone line search).
Returns the same fields `optimize_problem` reports, plus `bar_sigma2`.
"""
function optimize_emfree(
    model::BipartiteVarianceStableModel,
    stats::BipartiteGMRFStats,
    solver::EMFreeBlocks;
    verbose::Bool=false,
)
    fb = stats.free_blocks
    fb !== nothing || throw(ArgumentError("optimize_emfree requires error_blocks=:free statistics."))
    nf = stats.N_firms
    nw = stats.N_workers
    limit = rho_limit(model)
    ew = make_emfree_workspace(model, fb, nf, nw)
    Ωs = _init_blocks(fb)
    ψ = _emfree_ψ_from_θ(default_rho_start(limit), 0.7, 0.04, limit)
    # Fixed spectral-floor scale: the initial global mean trace (the model's
    # pinned barσ² reference), frozen for the whole EM — not a per-block trace.
    λ_lo = solver.eig_floor * (sum(tr(Ωi) for Ωi in Ωs) / stats.K)
    # Giant-firm guardrail (§5.2): the per-block m_i³ Cholesky and the dense
    # manager clique of S_z scale cubically; warn when a block is pathologically
    # large instead of silently paying it.
    mmax = maximum(fb.sizes; init = 0)
    mmax > 500 && @warn(
        "error_blocks=:free has a firm block of size $mmax; per-block Ω⁽ᵐ⁾³ work " *
        "and a dense manager clique of that size will dominate the EM."
    )

    rows = _block_rows(fb)
    nll_trace = Float64[]
    nll_curr = emfree_nll(model, fb, ew, Ωs, _emfree_θ_from_ψ(ψ, limit)..., nf, nw)
    push!(nll_trace, nll_curr)
    converged = false
    iterations = 0
    while iterations < solver.max_iter
        rho, sa, sz = _emfree_θ_from_ψ(ψ, limit)
        # E-step at (θ, Ω): posterior mode α̂ (the exact score needs it).
        alpha, _, _ = emfree_e_step(model, fb, ew, Ωs, rho, sa, sz, nf, nw)
        # One safeguarded score step on θ with Ω held fixed.
        gψ = _emfree_score(model, ew, alpha, ψ, limit)
        ψ = _emfree_theta_step(model, fb, ew, Ωs, ψ, gψ, limit, nf, nw)
        # M-step at the NEW θ: recompute S_ε(θ_new, Ω) so the block update is
        # consistent with the θ it accompanies (never the stale pre-step θ).
        rho2, sa2, sz2 = _emfree_θ_from_ψ(ψ, limit)
        _, Seps, _ = emfree_e_step(model, fb, ew, Ωs, rho2, sa2, sz2, nf, nw)
        for i in eachindex(Ωs)
            Ωs[i] = _floor_spectrum(Seps[i], λ_lo)
            C = cholesky(Symmetric(Ωs[i]); check = false)
            issuccess(C) || throw(ArgumentError(
                "error block $i (firm $(ew.f[first(rows[i])])) is not positive definite " *
                "after flooring (m_i = $(fb.sizes[i]), d_i = $(fb.distinct[i]))."
            ))
        end
        nll_new = emfree_nll(model, fb, ew, Ωs, rho2, sa2, sz2, nf, nw)
        push!(nll_trace, nll_new)
        iterations += 1
        Δ = nll_curr - nll_new
        scale = max(1.0, abs(nll_curr))
        if Δ >= solver.ftol * scale
            nll_curr = nll_new                        # meaningful improvement
        elseif Δ >= -solver.ftol * scale
            nll_curr = nll_new
            converged = true                           # flat within ftol → settled
            break
        else
            @warn "non-monotone EM step (nll increased by $(nll_new - nll_curr))."
            nll_curr = nll_new
        end
        verbose && @info "EMFreeBlocks iter $iterations: nll=$(nll_curr)"
    end

    rho, sa, sz = _emfree_θ_from_ψ(ψ, limit)
    converged || @warn(
        "EMFreeBlocks did not converge in $(solver.max_iter) iterations (nll change still above " *
        "ftol = $(solver.ftol)); the estimate may not be at a stationary point of the " *
        "marginal likelihood — treat it as non-settled regardless of the ρ value."
    )
    abs(rho) < 1e-2 && @warn(
        "Free-block fit landed at small |ρ| = $(abs(rho)) (flat likelihood region): " *
        "σ_a is identified only off the cross-firm network, so treat σ_a (and ρ) as " *
        "weakly identified rather than a precise point estimate (issue #112 §4, §7.7)."
    )
    bar_sigma2 = 0.0
    for (i, Ωi) in enumerate(Ωs)
        bar_sigma2 += tr(Ωi)
    end
    bar_sigma2 /= 0 < stats.K ? stats.K : 1
    return (
        rho = rho,
        sigma_a = sa * stats.y_std,
        sigma_z = sz * stats.y_std,
        sigma_epsilon = sqrt(max(bar_sigma2, 0.0)) * stats.y_std,
        rho_eps = nothing,
        beta = nothing,
        nll = nll_curr,
        converged = converged,
        iterations = iterations,
        obj_evals = iterations,
        optimization_time = 0.0,
        theta_unconstrained = ψ,
        model = model,
        stats = stats,
        omega = nothing,
        omega_sizes = nothing,
        bar_sigma2 = bar_sigma2,
        nll_trace = nll_trace,
    )
end
