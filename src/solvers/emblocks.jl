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
    # Constant across EM iterations and across θ: the node set S_i touched by
    # each firm block (sorted), and a structural pattern covering every
    # S_i × S_i, used to read M⁻¹ by selected inversion once per E-step.
    block_cols::Vector{Vector{Int}}
    selinv_pattern::SparseMatrixCSC{Float64,Int}
    # Per edge-level row: the two nodes it loads and their design values.
    edge_firm::Vector{Int}
    edge_worker::Vector{Int}
    edge_vf::Vector{Float64}
    edge_vw::Vector{Float64}
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
    f, w = _edge_nodes(fb.V, nf)
    vf, vw = _edge_values(fb.V, nf)
    cols, pattern = _block_selinv_pattern(fb, f, w, nf, nw)
    # edge_worker is stored as a *node* index (nf + worker), so the E-step never
    # has to re-derive the offset.
    return EMBlocksWorkspace(ws_Q, ws_M, f, cols, pattern, f, nf .+ w, vf, vw)
end

# Design values on each edge-level row's two nodes, in the same row order as
# `_edge_nodes`. error_blocks=:iw rejects match_id, so every row loads exactly
# one firm and one worker; the values are read rather than assumed to be 1.
function _edge_values(V::SparseMatrixCSC{Float64,Int}, nf::Int)
    K = size(V, 1)
    vf = zeros(Float64, K)
    vw = zeros(Float64, K)
    rows, cols, vals = findnz(V)
    for (r, c, v) in zip(rows, cols, vals)
        c <= nf ? (vf[r] = v) : (vw[r] = v)
    end
    return vf, vw
end

# The node set S_i = {firm_i} ∪ {workers of firm i} for every block, and a
# structural pattern covering every S_i × S_i. Built with explicit ones rather
# than reused from an assembled `P`, so an Ω that happens to produce a
# numerically zero entry can never drop a position out of the pattern.
function _block_selinv_pattern(fb::FirmBlockStats, f::Vector{Int}, w::Vector{Int},
                               nf::Int, nw::Int)
    n = nf + nw
    rows = _block_rows(fb)
    cols = Vector{Vector{Int}}(undef, length(rows))
    total = sum(m -> (m + 1)^2, fb.sizes)
    I = Vector{Int}(undef, 0); sizehint!(I, total)
    J = Vector{Int}(undef, 0); sizehint!(J, total)
    for i in eachindex(rows)
        r = rows[i]
        s = Vector{Int}(undef, length(r) + 1)
        s[1] = f[first(r)]
        for (a, t) in enumerate(r)
            s[a + 1] = nf + w[t]
        end
        unique!(sort!(s))
        cols[i] = s
        for p in s, q in s
            push!(I, p); push!(J, q)
        end
    end
    return cols, sparse(I, J, ones(Float64, length(I)), n, n, (x, y) -> 1.0)
end

# Posterior correction Aᵢ M⁻¹ Aᵢ' for one firm block, read off a selected
# inverse of M instead of solving against the whole factorization once per block
# row (issue #116: that cost K_tot global solves per E-step, each allocating a
# dense length-n vector — hours per iteration at n ≈ 2M).
#
# `Sig` is M⁻¹ read at the block pattern, `cols` the block's node set S_i.
# Σ_SS is a principal submatrix of M⁻¹ and therefore PD, so factoring it as
# LL' and returning (Aᵢ L)(Aᵢ L)' keeps the property the old triangular-solve
# form was written for: the result is a Gram product of an explicitly formed
# matrix, PSD to roundoff, with none of the cancellation a termwise
# selected-inverse sum would suffer. That matters because S_{ε,i} feeds
# `svals` → the Gamma rate `(δ + s)/2` → `log`, so a lost PSD is a NaN, not a
# slightly wrong number.
#
# If the small Cholesky fails (Σ_SS numerically indefinite at roundoff), fall
# back to the symmetrized congruence rather than failing the fit.
function _aptpa_local(
    Sig::SparseMatrixCSC{Float64,Int},
    cols::Vector{Int},
    rr::Vector{Int},
    ew::EMBlocksWorkspace,
)
    m = length(rr)
    s = length(cols)
    # Aᵢ restricted to the block's own nodes. `cols` is sorted, so the position
    # of each node is a binary search; every row loads exactly one firm and one
    # worker (match_id is rejected for error_blocks=:iw).
    A = zeros(Float64, m, s)
    for (a, t) in enumerate(rr)
        A[a, searchsortedfirst(cols, ew.edge_firm[t])] += ew.edge_vf[t]
        A[a, searchsortedfirst(cols, ew.edge_worker[t])] += ew.edge_vw[t]
    end

    Ssub = Matrix{Float64}(undef, s, s)
    for b in 1:s, a in 1:s
        Ssub[a, b] = Sig[cols[a], cols[b]]
    end
    Ssub .= 0.5 .* (Ssub .+ transpose(Ssub))

    C = cholesky(Symmetric(Ssub); check = false)
    if issuccess(C)
        W = A * Matrix(C.L)
        return W * transpose(W)
    end
    S = A * Ssub * transpose(A)
    return 0.5 .* (S .+ transpose(S))
end

# Reference implementation, kept for tests: the same quantity by one full
# triangular solve per block row. Correct but O(n) per row — see `_aptpa_local`,
# which replaced it in the E-step (issue #116).
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