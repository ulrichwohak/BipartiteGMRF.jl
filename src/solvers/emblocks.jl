# ═══════════════════════════════════════════════════════════════════════════
# Per-firm error blocks: shared EM scaffolding (issue #112)
#
# Data stay at edge level. Each firm's rows form one block whose error
# covariance is handled by whichever block solver is active (only the
# integrated-likelihood EMIWBlocks estimator in this tree). This file holds
# the solver-agnostic internals both the E-step and the reporting reuse:
# the edge design reader, the A'Ω⁻¹A assembly, the fixed-pattern workspace,
# and the Fisher-identity score.
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
function _block_rows(fb::FirmBlockStats)
    B = length(fb.sizes)
    rows = [Int[] for _ in 1:B]
    for s in eachindex(fb.block_of)
        push!(rows[fb.block_of[s]], s)
    end
    return rows
end

"""
    assemble_block_precision(fb, Ωs, nf, nw) -> (P_data, b)

Assemble the data term `A'Ω⁻¹A` (n×n sparse) and `A'Ω⁻¹y` (n) from the per-firm
blocks `Ωs` (a `Vector{Matrix{Float64}}`, one PD block per firm). `A = [D F]`
is the edge-level design; Ω is block-diagonal by firm, so the only structural
fill-in over the iid case is the manager-manager clique per firm. No global Ω
is materialized. Throws if a block Cholesky fails (non-PD guardrail).
"""
function assemble_block_precision(fb::FirmBlockStats, Ωs::Vector{Matrix{Float64}}, nf::Int, nw::Int)
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
Workspace for the block-solver EM. `ws_Q` factors the prior precision `K`
(pattern fixed across ρ ≠ 0); `ws_M` factors the posterior `M = K + A'Ω⁻¹A`
(pattern fixed as the full per-firm block pattern — see the plan notes). `f`
holds the firm node of each edge-level row (for the PD guardrail).
"""
struct EMBlocksWorkspace
    ws_Q::GaussianMarkovRandomFields.GMRFWorkspace
    ws_M::GaussianMarkovRandomFields.GMRFWorkspace
    f::Vector{Int}
end

function _init_blocks(fb::FirmBlockStats)
    return [Matrix{Float64}(I, m, m) .+ 0.5 .* (ones(m, m) .- Matrix{Float64}(I, m, m))
            for m in fb.sizes]
end

# Values of `Q + P` read onto `ws`'s fixed pattern, as an nzval vector for
# `update_precision_values!`. Keeps the workspace pattern stable even when a
# diagonal Ω (or ρ → 0) would otherwise drop entries from `Q + P` and trip the
# exact-pattern check in `update_precision!`.
function _align_to_ws(Q::SparseMatrixCSC{Float64,Int}, P::Union{Nothing,SparseMatrixCSC{Float64,Int}},
                      ws::GaussianMarkovRandomFields.GMRFWorkspace)
    n = size(Q, 1)
    vals = zeros(Float64, nnz(ws.Q))
    for j in 1:n
        for p in nzrange(ws.Q, j)
            i = ws.Q.rowval[p]
            vals[p] = Q[i, j] + (P === nothing ? 0.0 : P[i, j])
        end
    end
    return vals
end

function make_em_blocks_workspace(model::AbstractBipartiteModel, fb::FirmBlockStats, nf::Int, nw::Int)
    # Seed ω_M with the FULL per-firm block pattern (reference blocks carry full
    # within-firm off-diagonal support), so the symbolic factorization covers
    # the manager-manager fill-in that a diagonal (iid) seed would drop.
    P0, _ = assemble_block_precision(fb, _init_blocks(fb), nf, nw)
    rho_ref = min(0.1, 0.5 * rho_limit(model)) + eps(Float64)
    Q0 = model_precision(model, rho_ref, 1.0, 1.0)
    ws_Q = GaussianMarkovRandomFields.GMRFWorkspace(Q0)
    ws_M = GaussianMarkovRandomFields.GMRFWorkspace(Q0 + P0)
    f, _ = _edge_nodes(fb.V, nf)
    return EMBlocksWorkspace(ws_Q, ws_M, f)
end

# Posterior correction Aᵢ M⁻¹ Aᵢ' for the block's edge-level design rows Vᵢ,
# computed as Vᵢ · (M⁻¹ Vᵢ') by triangular solves. This Gram form avoids the
# cancellation of the 4-term selected-inverse sum when M is ill-conditioned;
# each entry is an inner product of *solved* vectors, so the result stays PSD
# to roundoff.
function _aptpa(ws::GaussianMarkovRandomFields.GMRFWorkspace, Vi::SparseMatrixCSC{Float64,Int})
    m = size(Vi, 1)
    n = size(Vi, 2)
    X = Matrix{Float64}(undef, n, m)
    for a in 1:m
        X[:, a] = GaussianMarkovRandomFields.workspace_solve(ws, Vector(Vi[a, :]))
    end
    return Matrix(Vi * X)
end

# ─── Hyperparameter codec and score ────────────────────────────────────────

function _em_blocks_θ_from_ψ(ψ::Vector{Float64}, limit::Float64)
    ρ = limit * tanh(clamp(ψ[1], -25.0, 25.0))
    # tanh saturates to 1.0 in Float64 well before the ±25 clamp, which would
    # map ρ to exactly ±limit and trip the strict |ρ| < limit domain check.
    # Stay strictly interior.
    ρmax = limit * (1 - 1e-10)
    return clamp(ρ, -ρmax, ρmax), exp(ψ[2]), exp(ψ[3])
end

function _em_blocks_ψ_from_θ(rho::Float64, sa::Float64, sz::Float64, limit::Float64)
    return [atanh(rho / limit), log(sa), log(sz)]
end

"""
Exact marginal score (Fisher identity) of the NLL w.r.t. `(ρ, σ_a, σ_z)`, in
ψ-space. The complete-data score `∂ℓ/∂θ_k = ½ tr(∂K/∂θ_k (K⁻¹ − S_α))` equals
the marginal score at any interior point; with `S_α = α̂α̂' + M⁻¹` this reads
entirely off `selinv_dot` of the already-factored `K` and `M` workspaces plus
the penalized-GLS mode `α̂`. `∂K/∂θ` is finite-differenced (K is cheap to
build). Returns the NLL gradient mapped to ψ through the codec Jacobian.
"""
function _em_blocks_score(
    model::BipartiteVarianceStableModel,
    ew::EMBlocksWorkspace,
    alpha::Vector{Float64},
    ψ::Vector{Float64},
    limit::Float64,
)
    rho, sa, sz = _em_blocks_θ_from_ψ(ψ, limit)
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