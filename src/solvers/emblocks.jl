# ═══════════════════════════════════════════════════════════════════════════
# Per-firm error blocks: shared EM scaffolding (issue #112)
#
# Observations are raw rows, or matches when match_id groups them (issue
# #120). Each firm's observations form one block whose error covariance is
# handled by whichever block solver is active (only the integrated-likelihood
# EMIWBlocks estimator in this tree). This file holds the solver-agnostic
# internals both the E-step and the reporting reuse: the row-support reader,
# the A'Ω⁻¹A assembly, the fixed-pattern workspace, and the Fisher-identity
# score.
# ═══════════════════════════════════════════════════════════════════════════

# Node columns, design values, and firm node of each observation row, read
# once from the design `V`. A raw row has exactly two nonzeros (firm 1.0,
# worker 1.0); a match-collapsed row has one firm node (weight `1/F = 1`,
# matches span a single firm) and one or more worker nodes (weight `1/M`
# each). All downstream consumers iterate the (node, value) lists rather
# than assuming a per-row (firm, worker) pair.
function _row_supports(V::SparseMatrixCSC{Float64,Int}, nf::Int)
    K = size(V, 1)
    f = zeros(Int, K)
    nodes = [Int[] for _ in 1:K]
    vals = [Float64[] for _ in 1:K]
    rws, cls, vs = findnz(V)
    for (r, c, v) in zip(rws, cls, vs)
        push!(nodes[r], c)
        push!(vals[r], v)
        c <= nf && (f[r] = c)
    end
    return f, nodes, vals
end

# Observation gaps: gap i lists the observation row indices whose block_of is i.
function _block_rows(fb::FirmBlockStats)
    B = length(fb.sizes)
    rows = [Int[] for _ in 1:B]
    for s in eachindex(fb.block_of)
        push!(rows[fb.block_of[s]], s)
    end
    return rows
end

"""
    assemble_block_precision(fb, Ωs, nf, nw, supports=nothing) -> (P_data, b)

Assemble the data term `A'Ω⁻¹A` (n×n sparse) and `A'Ω⁻¹y` (n) from the per-firm
blocks `Ωs` (a `Vector{Matrix{Float64}}`, one PD block per firm). `A` is the
observation-level design (raw rows, or match-collapsed rows with `1/M`-weighted
workers); Ω is block-diagonal by firm, so the only structural fill-in over the
iid case is the manager-manager clique per firm. The accumulation iterates each
row's actual (node, value) support, so weighted multi-worker rows contribute
`G[a,c]·v_a·v_c` at every node pair. No global Ω is materialized. Throws if a
block Cholesky fails (non-PD guardrail).
"""
function assemble_block_precision(
    fb::FirmBlockStats,
    Ωs::Vector{Matrix{Float64}},
    nf::Int,
    nw::Int,
    supports::Union{Nothing,Tuple{Vector{Int},Vector{Vector{Int}},Vector{Vector{Float64}}}}=nothing,
)
    n = nf + nw
    # The row supports are Ω-independent; the EM caches them on the workspace
    # and passes them in, so the per-iteration call allocates no per-row
    # vectors.
    f, nodes, vals = supports === nothing ? _row_supports(fb.V, nf) : supports
    rows = _block_rows(fb)
    length(Ωs) == length(rows) ||
        throw(ArgumentError("one Ω per firm block expected."))

    I = Int[]
    J = Int[]
    V = Float64[]
    total = sum(r -> sum(t -> length(nodes[t]), r; init = 0)^2, rows; init = 0)
    sizehint!(I, total)
    sizehint!(J, total)
    sizehint!(V, total)
    b = zeros(Float64, n)

    for i in eachindex(rows)
        r = rows[i]
        ms = fb.sizes[i]
        C = cholesky(Symmetric(Ωs[i]); check = false)
        issuccess(C) || throw(ArgumentError(
            "error block $i (firm $(f[first(r)])) is not positive definite (m=$ms)."))
        G = inv(C)                       # Ω_i⁻¹ (dense, block-local ordering)
        yi = fb.y[r]
        Gy = G * yi

        for a in 1:ms
            ta = r[a]
            # A'Ω⁻¹y on row a's support
            for (pa, va) in zip(nodes[ta], vals[ta])
                b[pa] += va * Gy[a]
            end
            # A'Ω⁻¹A: every node pair of rows (a, c) at weight G[a,c]·v_a·v_c
            for c in 1:ms
                g = G[a, c]
                tc = r[c]
                for (pa, va) in zip(nodes[ta], vals[ta])
                    gva = g * va
                    for (pc, vc) in zip(nodes[tc], vals[tc])
                        push!(I, pa); push!(J, pc); push!(V, gva * vc)
                    end
                end
            end
        end
    end
    return sparse(I, J, V, n, n), b
end

# ─── EM scaffolding ────────────────────────────────────────────────────────

"""
Workspace for the block-solver EM. `ws_Q` factors the prior precision `K`
(pattern fixed across ρ ≠ 0); `ws_M` factors the posterior `M = K + A'Ω⁻¹A`
(pattern fixed as the full per-firm block pattern — see the plan notes). `f`
holds the firm node of each observation row (for the PD guardrail).
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
    # Per observation row: the nodes it loads and their design values (one
    # firm plus one or more 1/M-weighted workers under match grouping).
    row_nodes::Vector{Vector{Int}}
    row_vals::Vector{Vector{Float64}}
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
    n = nf + nw
    f, nodes, vals = _row_supports(fb.V, nf)
    cols, pattern = _block_selinv_pattern(fb, nodes, n)
    # Seed ω_M with the FULL per-firm block pattern, taken STRUCTURALLY as the
    # union of Q's pattern and the S_i × S_i cliques. Numeric seeding
    # (Q0 + P0) is not safe here: with weighted multi-worker rows an
    # off-diagonal entry of P0 can cancel analytically (e.g. chained
    # two-worker matches give 1/4 − 1/4 at the shared-worker position), and
    # sparse `+` drops exact zeros — silently losing the position from the
    # fixed symbolic pattern while the runtime M is nonzero there. The
    # all-ones union cannot cancel; the numeric values are the reference
    # Q0 + P0 read onto it (explicit zeros kept).
    P0, _ = assemble_block_precision(fb, _init_blocks(fb), nf, nw, (f, nodes, vals))
    rho_ref = min(0.1, 0.5 * rho_limit(model)) + eps(Float64)
    Q0 = model_precision(model, rho_ref, 1.0, 1.0)
    ws_Q = GaussianMarkovRandomFields.GMRFWorkspace(Q0)
    mark(A) = SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), ones(Float64, nnz(A)))
    union_pat = SparseMatrixCSC{Float64,Int}(mark(Q0) + mark(pattern))
    nzv = zeros(Float64, nnz(union_pat))
    for j in 1:n
        for p in nzrange(union_pat, j)
            i = union_pat.rowval[p]
            nzv[p] = Q0[i, j] + P0[i, j]
        end
    end
    M0 = SparseMatrixCSC(n, n, union_pat.colptr, union_pat.rowval, nzv)
    ws_M = GaussianMarkovRandomFields.GMRFWorkspace(M0)
    return EMBlocksWorkspace(ws_Q, ws_M, f, cols, pattern, nodes, vals)
end

# The node set S_i = {firm_i} ∪ {workers of the block's observations} for
# every block, and a structural pattern covering every S_i × S_i. Built with
# explicit ones rather than reused from an assembled `P`, so an Ω that happens
# to produce a numerically zero entry can never drop a position out of the
# pattern.
function _block_selinv_pattern(fb::FirmBlockStats, nodes::Vector{Vector{Int}}, n::Int)
    rows = _block_rows(fb)
    cols = Vector{Vector{Int}}(undef, length(rows))
    for i in eachindex(rows)
        s = Int[]
        for t in rows[i]
            append!(s, nodes[t])
        end
        unique!(sort!(s))
        cols[i] = s
    end
    total = sum(s -> length(s)^2, cols; init = 0)
    I = Vector{Int}(undef, 0); sizehint!(I, total)
    J = Vector{Int}(undef, 0); sizehint!(J, total)
    for s in cols
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
    # of each node is a binary search; each row contributes its full (node,
    # value) support — one firm plus one or more weighted workers under match
    # grouping.
    A = zeros(Float64, m, s)
    for (a, t) in enumerate(rr)
        for (p, v) in zip(ew.row_nodes[t], ew.row_vals[t])
            A[a, searchsortedfirst(cols, p)] += v
        end
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