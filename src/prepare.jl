# Design-product construction shared by suffstats and the rho_eps
# re-estimation path.

function build_V_stats(
    f_rows::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    n_firms::Int,
    n_workers::Int,
)
    k = length(y)
    length(f_rows) == k || throw(ArgumentError("f_rows length does not match y."))
    length(w_cols) == k || throw(ArgumentError("w_cols length does not match y."))

    proj_f = zeros(Float64, n_firms)
    proj_w = zeros(Float64, n_workers)
    cnt_f = zeros(Float64, n_firms)
    cnt_w = zeros(Float64, n_workers)

    @inbounds for i in 1:k
        f = f_rows[i]
        w = w_cols[i]
        val = y[i]
        cnt_f[f] += 1.0
        cnt_w[w] += 1.0
        proj_f[f] += val
        proj_w[w] += val
    end

    A_obs = sparse(f_rows, w_cols, ones(Float64, k), n_firms, n_workers)
    FF = spdiagm(0 => cnt_f)
    WW = spdiagm(0 => cnt_w)
    VtV = [FF A_obs; copy(transpose(A_obs)) WW]
    return DesignStats(
        VtV,
        vcat(proj_f, proj_w),
        dot(y, y),
        A_obs,
        copy(transpose(A_obs)),
        FF,
        WW,
    )
end

function build_weighted_V_stats(
    f_rows::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    weights::Vector{Float64},
    n_firms::Int,
    n_workers::Int,
)
    k = length(y)
    length(f_rows) == k || throw(ArgumentError("f_rows length does not match y."))
    length(w_cols) == k || throw(ArgumentError("w_cols length does not match y."))
    length(weights) == k || throw(ArgumentError("weights length does not match y."))

    proj_f = zeros(Float64, n_firms)
    proj_w = zeros(Float64, n_workers)
    cnt_f = zeros(Float64, n_firms)
    cnt_w = zeros(Float64, n_workers)
    ydot = 0.0

    @inbounds for i in 1:k
        f = f_rows[i]
        w = w_cols[i]
        wi = weights[i]
        val = y[i]
        cnt_f[f] += wi
        cnt_w[w] += wi
        proj_f[f] += wi * val
        proj_w[w] += wi * val
        ydot += wi * val * val
    end

    A_obs = sparse(f_rows, w_cols, weights, n_firms, n_workers)
    FF = spdiagm(0 => cnt_f)
    WW = spdiagm(0 => cnt_w)
    VtV = [FF A_obs; copy(transpose(A_obs)) WW]
    return DesignStats(
        VtV,
        vcat(proj_f, proj_w),
        ydot,
        A_obs,
        copy(transpose(A_obs)),
        FF,
        WW,
    )
end

function build_match_weight_stats(
    f_rows::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    T::Vector{Int},
    n_firms::Int,
    n_workers::Int,
    rho_eps::Float64,
)
    weights = effective_match_weights(T, rho_eps)
    design = build_weighted_V_stats(f_rows, w_cols, y, weights, n_firms, n_workers)
    weight_stats = WeightStats(
        sum(log, weights),
        sum(weights),
        sum(weights ./ Float64.(T)),
        mean(weights),
        maximum(weights),
    )
    return design, weight_stats
end

"""
Build design products for match-grouped observations. Each match `s`
has a design row that averages `1/F_s` over its firms and `1/M_s` over
its workers. Returns a `DesignStats` with (potentially) off-diagonal
`FF` and `WW` blocks.
"""
function build_match_V_stats(
    f_rows::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    match_ids::Vector{Int},
    n_firms::Int,
    n_workers::Int,
)
    k = length(y)

    # Group edges by match id → per-match firm set, worker set, outcome
    match_map = Dict{Int,Int}()        # match_id → position
    match_firms = Vector{Vector{Int}}()
    match_workers = Vector{Vector{Int}}()
    match_y = Float64[]

    for i in 1:k
        mid = match_ids[i]
        pos = get!(match_map, mid) do
            push!(match_firms, Int[])
            push!(match_workers, Int[])
            push!(match_y, y[i])
            length(match_y)
        end
        push!(match_firms[pos], f_rows[i])
        push!(match_workers[pos], w_cols[i])
    end
    n_matches = length(match_y)

    # COO accumulators for sparse blocks
    ff_i = Int[]; ff_j = Int[]; ff_v = Float64[]
    ww_i = Int[]; ww_j = Int[]; ww_v = Float64[]
    a_i  = Int[]; a_j  = Int[]; a_v  = Float64[]
    proj_f = zeros(Float64, n_firms)
    proj_w = zeros(Float64, n_workers)
    ydot = 0.0

    for s in 1:n_matches
        firms_s = unique(match_firms[s])
        workers_s = unique(match_workers[s])
        F_s = length(firms_s)
        M_s = length(workers_s)
        ys = match_y[s]

        # FF block: 1/F_s^2 for all (i, i') in firms_s × firms_s
        ff_val = 1.0 / F_s^2
        for fi in firms_s, fj in firms_s
            push!(ff_i, fi); push!(ff_j, fj); push!(ff_v, ff_val)
        end

        # WW block: 1/M_s^2 for all (m, m') in workers_s × workers_s
        ww_val = 1.0 / M_s^2
        for wi in workers_s, wj in workers_s
            push!(ww_i, wi); push!(ww_j, wj); push!(ww_v, ww_val)
        end

        # Cross block: 1/(F_s * M_s) for all (i, m)
        a_val = 1.0 / (F_s * M_s)
        for fi in firms_s, wj in workers_s
            push!(a_i, fi); push!(a_j, wj); push!(a_v, a_val)
        end

        # V'y
        for fi in firms_s
            proj_f[fi] += ys / F_s
        end
        for wj in workers_s
            proj_w[wj] += ys / M_s
        end

        # y'y
        ydot += ys^2
    end

    FF = sparse(ff_i, ff_j, ff_v, n_firms, n_firms)
    WW = sparse(ww_i, ww_j, ww_v, n_workers, n_workers)
    A_obs = sparse(a_i, a_j, a_v, n_firms, n_workers)
    VtV = [FF A_obs; copy(transpose(A_obs)) WW]

    return DesignStats(
        VtV,
        vcat(proj_f, proj_w),
        ydot,
        A_obs,
        copy(transpose(A_obs)),
        FF,
        WW,
    )
end

# ─── Mean-structure statistics ────────────────────────────────────────────

function build_mean_stats(
    f_rows::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    X::Matrix{Float64},
    n_firms::Int,
    n_workers::Int,
)
    k = length(y)
    p = size(X, 2)
    n = n_firms + n_workers
    VtX = zeros(Float64, n, p)
    Xty = zeros(Float64, p)

    @inbounds for i in 1:k
        f = f_rows[i]
        w = w_cols[i]
        for j in 1:p
            xij = X[i, j]
            VtX[f, j] += xij
            VtX[n_firms + w, j] += xij
            Xty[j] += xij * y[i]
        end
    end

    XtX = X' * X
    return MeanStats(VtX, XtX, Xty, p)
end

function build_weighted_mean_stats(
    f_rows::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    X::Matrix{Float64},
    weights::Vector{Float64},
    n_firms::Int,
    n_workers::Int,
)
    k = length(y)
    p = size(X, 2)
    n = n_firms + n_workers
    VtX = zeros(Float64, n, p)
    Xty = zeros(Float64, p)
    XtX = zeros(Float64, p, p)

    @inbounds for i in 1:k
        f = f_rows[i]
        w = w_cols[i]
        wi = weights[i]
        for j in 1:p
            wxij = wi * X[i, j]
            VtX[f, j] += wxij
            VtX[n_firms + w, j] += wxij
            Xty[j] += wxij * y[i]
            for j2 in 1:p
                XtX[j, j2] += wxij * X[i, j2]
            end
        end
    end

    return MeanStats(VtX, XtX, Xty, p)
end

function build_match_mean_stats(
    f_rows::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    X::Matrix{Float64},
    match_ids::Vector{Int},
    n_firms::Int,
    n_workers::Int,
)
    k = length(y)
    p = size(X, 2)
    n = n_firms + n_workers
    VtX = zeros(Float64, n, p)
    Xty = zeros(Float64, p)

    # Group by match
    match_map = Dict{Int,Int}()
    match_firms = Vector{Vector{Int}}()
    match_workers = Vector{Vector{Int}}()
    match_y = Float64[]
    match_x = Vector{Vector{Float64}}()

    for i in 1:k
        mid = match_ids[i]
        pos = get!(match_map, mid) do
            push!(match_firms, Int[])
            push!(match_workers, Int[])
            push!(match_y, y[i])
            push!(match_x, X[i, :])
            length(match_y)
        end
        push!(match_firms[pos], f_rows[i])
        push!(match_workers[pos], w_cols[i])
    end

    XtX = zeros(Float64, p, p)
    for s in eachindex(match_y)
        firms_s = unique(match_firms[s])
        workers_s = unique(match_workers[s])
        F_s = length(firms_s)
        M_s = length(workers_s)
        xs = match_x[s]

        for fi in firms_s
            for j in 1:p
                VtX[fi, j] += xs[j] / F_s
            end
        end
        for wj in workers_s
            for j in 1:p
                VtX[n_firms + wj, j] += xs[j] / M_s
            end
        end

        for j in 1:p
            Xty[j] += xs[j] * match_y[s]
            for j2 in 1:p
                XtX[j, j2] += xs[j] * xs[j2]
            end
        end
    end

    return MeanStats(VtX, XtX, Xty, p)
end

function build_match_weight_mean_stats(
    f_rows::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    X::Matrix{Float64},
    T::Vector{Int},
    n_firms::Int,
    n_workers::Int,
    rho_eps::Float64,
)
    weights = effective_match_weights(T, rho_eps)
    return build_weighted_mean_stats(f_rows, w_cols, y, X, weights, n_firms, n_workers)
end

"""
Collapse repeated (firm, worker) observations to unique edges, preserving
first-appearance order. Returns parallel vectors of firm index, worker index,
observation count `T`, edge mean of `y`, and within-edge sum of squares.
"""
function collapse_edges(
    f_rows::AbstractVector{<:Integer},
    w_cols::AbstractVector{<:Integer},
    y::AbstractVector{Float64},
)
    index = Dict{Tuple{Int,Int},Int}()
    f = Int[]
    w = Int[]
    T = Int[]
    y_sum = Float64[]
    y_sq = Float64[]
    for k in eachindex(f_rows)
        key = (Int(f_rows[k]), Int(w_cols[k]))
        j = get!(index, key) do
            push!(f, key[1])
            push!(w, key[2])
            push!(T, 0)
            push!(y_sum, 0.0)
            push!(y_sq, 0.0)
            length(f)
        end
        T[j] += 1
        y_sum[j] += y[k]
        y_sq[j] += y[k]^2
    end
    y_mean = y_sum ./ T
    ssw = [max(y_sq[j] - y_sum[j]^2 / T[j], 0.0) for j in eachindex(T)]
    return (f=f, w=w, T=T, y_mean=y_mean, ssw=ssw)
end

# ─── General correlated errors: sigma_eps^2 R, block-sparse PD R ─────────────

"""
Observation-level design rows: one row per input row, or one row per match
when `match_ids` is given (each distinct firm weighted `1/F`, each distinct
worker `1/M`, as in `build_match_V_stats`). Returns the sparse `V` (K × n),
the outcome vector, and `src`, mapping each observation to its first input
row.
"""
function observation_rows(
    f_obs::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    match_ids::Union{Nothing,Vector{Int}},
    n_firms::Int,
    n_workers::Int,
)
    n = n_firms + n_workers
    Vi = Int[]; Vj = Int[]; Vv = Float64[]
    yv = Float64[]; src = Int[]
    if match_ids === nothing
        for i in eachindex(y)
            push!(yv, y[i]); push!(src, i)
            k = length(yv)
            push!(Vi, k); push!(Vj, f_obs[i]); push!(Vv, 1.0)
            push!(Vi, k); push!(Vj, n_firms + w_cols[i]); push!(Vv, 1.0)
        end
    else
        pos = Dict{Int,Int}()
        firms_of = Vector{Vector{Int}}()
        workers_of = Vector{Vector{Int}}()
        for i in eachindex(y)
            s = get!(pos, match_ids[i]) do
                push!(firms_of, Int[]); push!(workers_of, Int[])
                push!(yv, y[i]); push!(src, i)
                length(yv)
            end
            push!(firms_of[s], f_obs[i])
            push!(workers_of[s], w_cols[i])
        end
        for s in eachindex(yv)
            fs = unique(firms_of[s]); ws = unique(workers_of[s])
            for f in fs
                push!(Vi, s); push!(Vj, f); push!(Vv, 1.0 / length(fs))
            end
            for w in ws
                push!(Vi, s); push!(Vj, n_firms + w); push!(Vv, 1.0 / length(ws))
            end
        end
    end
    return sparse(Vi, Vj, Vv, length(yv), n), yv, src
end

"""
Collapse a per-row attribute to one value per observation, in the same
first-appearance order [`observation_rows`](@ref) assigns, requiring the
attribute to be constant within each match. Used by the structured error
models to read the per-match firm (each match must span a single firm) and
the per-match `edge_index`.
"""
function per_observation_value(
    vals::Vector{Int},
    match_ids::Vector{Int},
    what::AbstractString,
)
    pos = Dict{Int,Int}()
    out = Int[]
    for i in eachindex(vals)
        s = get!(pos, match_ids[i]) do
            push!(out, vals[i])
            length(out)
        end
        out[s] == vals[i] || throw(ArgumentError(
            "$what must be constant within a match; match $(match_ids[i]) " *
            "carries $(out[s]) and $(vals[i])."))
    end
    return out
end

"""
    build_correlated_V_stats(f_obs, w_obs, y, match_ids, R, n_firms, n_workers)

Design products under a general error covariance `sigma_eps^2 R`, where `R` is
a sparse symmetric matrix over observations whose connected blocks (read off
the sparsity pattern) are each positive definite. The blocks are otherwise
arbitrary — any correlation pattern, any within-block heteroskedasticity. `R`'s
overall scale is pinned here by rescaling to `tr(R) = K`, so `sigma_eps^2`
keeps a fixed meaning as the mean error variance; the caller's scale is
irrelevant.

Returns `(design, logdet_R, K, aux)`. `design` carries `V'R⁻¹V`, `V'R⁻¹y` and
`y'R⁻¹y` in the usual `DesignStats` fields, so every downstream consumer —
`M = Q + λV'V`, the quadratic form, the Hutchinson blocks — works unchanged;
`logdet_R` enters the likelihood through `WeightStats.log_weight_sum` (as
`-logdet_R`, matching the sign convention `2·NLL ⊃ -log_weight_sum`). `aux`
holds `(V, Rinv, y, src)` so mean-structure products can be R-weighted without
rebuilding; `src` maps each observation to its first input row.

With `match_ids`, edges sharing an id form one observation exactly as in
`build_match_V_stats` (each distinct firm weighted `1/F`, each distinct worker
`1/M`); `R` is then read at each match's first row, and entries between rows
of one match are dropped with the duplicate rows.
"""
function build_correlated_V_stats(
    f_obs::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    match_ids::Union{Nothing,Vector{Int}},
    R_rows::SparseMatrixCSC{Float64,Int},
    n_firms::Int,
    n_workers::Int,
)
    n = n_firms + n_workers
    V, yv, src = observation_rows(f_obs, w_cols, y, match_ids, n_firms, n_workers)
    K = length(yv)

    # Observation-space R: representative rows under grouping, then the scale
    # normalization that keeps sigma_eps^2 interpretable.
    R = match_ids === nothing ? R_rows : SparseMatrixCSC{Float64,Int}(R_rows[src, src])
    trR = sum(R[i, i] for i in 1:K)
    trR > 0 || throw(ArgumentError("error_cov must have positive diagonal entries."))
    R = R .* (K / trR)

    # Blocks = connected components of the sparsity pattern (union-find).
    parent = collect(1:K)
    function findroot(x::Int)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    rows = rowvals(R)
    for j in 1:K
        for ptr in nzrange(R, j)
            i = rows[ptr]
            i == j && continue
            ri, rj = findroot(i), findroot(j)
            ri != rj && (parent[ri] = rj)
        end
    end
    blocks = Dict{Int,Vector{Int}}()
    for i in 1:K
        push!(get!(blocks, findroot(i), Int[]), i)
    end

    Ri = Int[]; Rj = Int[]; Rv = Float64[]
    logdet_R = 0.0
    for (_, idx) in blocks
        if length(idx) == 1
            r = R[idx[1], idx[1]]
            r > 0 || throw(ArgumentError("error_cov has a non-positive diagonal entry."))
            logdet_R += log(r)
            push!(Ri, idx[1]); push!(Rj, idx[1]); push!(Rv, 1.0 / r)
            continue
        end
        B = Matrix(R[idx, idx])
        C = cholesky(Symmetric(B); check=false)
        issuccess(C) || throw(ArgumentError(
            "error_cov has a block of $(length(idx)) observations that is not positive definite."))
        logdet_R += logdet(C)
        Binv = inv(C)
        for a in eachindex(idx), b in eachindex(idx)
            push!(Ri, idx[a]); push!(Rj, idx[b]); push!(Rv, Binv[a, b])
        end
    end
    Rinv = sparse(Ri, Rj, Rv, K, K)

    VtRV = SparseMatrixCSC{Float64,Int}(transpose(V) * Rinv * V)
    Ry = Rinv * yv
    projected = Vector{Float64}(transpose(V) * Ry)
    ydot = dot(yv, Ry)
    FF = SparseMatrixCSC{Float64,Int}(VtRV[1:n_firms, 1:n_firms])
    WW = SparseMatrixCSC{Float64,Int}(VtRV[(n_firms+1):n, (n_firms+1):n])
    A_obs = SparseMatrixCSC{Float64,Int}(VtRV[1:n_firms, (n_firms+1):n])
    design = DesignStats(VtRV, projected, ydot, A_obs,
                         SparseMatrixCSC{Float64,Int}(copy(transpose(A_obs))), FF, WW)
    return design, logdet_R, K, (V = V, Rinv = Rinv, y = yv, src = src)
end

"""
R-weighted mean-structure products for the correlated error model. Storing
`V'R⁻¹X`, `X'R⁻¹X` and `X'R⁻¹y` makes `mean_profile_correction` correct
without modification: its `c` and `G` become the GLS versions automatically.
"""
function build_correlated_mean_stats(aux, X_rows::Matrix{Float64})
    RX = aux.Rinv * X_rows
    return MeanStats(
        Matrix{Float64}(transpose(aux.V) * RX),
        Matrix{Float64}(transpose(X_rows) * RX),
        Vector{Float64}(transpose(X_rows) * (aux.Rinv * aux.y)),
        size(X_rows, 2),
    )
end

# ─── Group-robust errors: free per-size-class variance of group means ────────

"""
`A`'s nonzero values scattered onto `P`'s sparsity pattern (which must contain
`A`'s). Returns a vector aligned with `nonzeros(P)`.
"""
function align_nzvals(P::SparseMatrixCSC{Float64,Int}, A::SparseMatrixCSC{Float64,Int})
    out = zeros(Float64, nnz(P))
    Prv = rowvals(P); Arv = rowvals(A); Anz = nonzeros(A)
    for j in 1:size(A, 2)
        lo = P.colptr[j]; hi = P.colptr[j + 1] - 1
        for ptr in nzrange(A, j)
            i = Arv[ptr]
            pos = searchsortedfirst(view(Prv, lo:hi), i) + lo - 1
            (pos <= hi && Prv[pos] == i) ||
                throw(ArgumentError("internal: class pattern not contained in the pooled pattern."))
            out[pos] += Anz[ptr]
        end
    end
    return out
end

"""
    build_grouped_V_stats(f_obs, w_obs, y, match_ids, group_obs, cap, n_firms, n_workers)

Design products for group-robust errors: observations sharing a group id
(typically the firm) are collapsed to their mean — design rows averaged,
outcomes averaged — so that an arbitrary unknown PD error covariance within
the group enters the likelihood only through the scalar variance of the group
mean. That scalar is absorbed by one free parameter per group-size class,
estimated inside the MLE; groups of size `cap` or larger share the top class.

Returns `(design, K, ec, aux)`: the pooled `DesignStats` over collapsed
observations, the group count, the per-class blocks for [`ErrorClassStats`](@ref)
(sans mean stats), and `(G, Vg, yg, class_idx, src)` for building R-weighted
mean-structure products. Classes are ordered by ascending size; the smallest
class present is the one pinned at `omega = 1`.

Composes with `match_ids` (rows of one match must share a group id).
"""
function build_grouped_V_stats(
    f_obs::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    match_ids::Union{Nothing,Vector{Int}},
    group_obs::Vector{Int},
    cap::Int,
    n_firms::Int,
    n_workers::Int,
)
    if match_ids !== nothing
        match_group = Dict{Int,Int}()
        for i in eachindex(match_ids)
            prev = get!(match_group, match_ids[i], group_obs[i])
            prev == group_obs[i] || throw(ArgumentError(
                "match $(match_ids[i]) spans more than one error group."))
        end
    end
    V, yv, src = observation_rows(f_obs, w_cols, y, match_ids, n_firms, n_workers)
    n_units = length(yv)
    n = n_firms + n_workers

    gid = group_obs[src]
    gpos = Dict{Int,Int}()
    members = Vector{Vector{Int}}()
    for s in 1:n_units
        g = get!(gpos, gid[s]) do
            push!(members, Int[])
            length(members)
        end
        push!(members[g], s)
    end
    K = length(members)
    ksz = length.(members)

    Gi = Int[]; Gj = Int[]; Gv = Float64[]
    for g in 1:K, s in members[g]
        push!(Gi, g); push!(Gj, s); push!(Gv, 1.0 / ksz[g])
    end
    G = sparse(Gi, Gj, Gv, K, n_units)
    Vg = SparseMatrixCSC{Float64,Int}(G * V)
    yg = Vector{Float64}(G * yv)

    labels = min.(ksz, cap)
    sizes = sort(unique(labels))
    C = length(sizes)
    class_of = Dict(sz => c for (c, sz) in enumerate(sizes))
    class_idx = [Int[] for _ in 1:C]
    for g in 1:K
        push!(class_idx[class_of[labels[g]]], g)
    end

    class_vtv = Vector{SparseMatrixCSC{Float64,Int}}(undef, C)
    projected = zeros(Float64, n, C)
    ydot = zeros(Float64, C)
    for c in 1:C
        Vc = Vg[class_idx[c], :]
        yc = yg[class_idx[c]]
        class_vtv[c] = SparseMatrixCSC{Float64,Int}(transpose(Vc) * Vc)
        projected[:, c] = transpose(Vc) * yc
        ydot[c] = sum(abs2, yc)
    end
    # Pooled pattern as the SUM of the class matrices: entries of V are
    # nonnegative, so nothing cancels and every class pattern is contained.
    VtV = reduce(+, class_vtv)
    vtv_nzvals = zeros(Float64, nnz(VtV), C)
    for c in 1:C
        vtv_nzvals[:, c] = align_nzvals(VtV, class_vtv[c])
    end

    FF = SparseMatrixCSC{Float64,Int}(VtV[1:n_firms, 1:n_firms])
    WW = SparseMatrixCSC{Float64,Int}(VtV[(n_firms+1):n, (n_firms+1):n])
    A_obs = SparseMatrixCSC{Float64,Int}(VtV[1:n_firms, (n_firms+1):n])
    design = DesignStats(VtV, vec(sum(projected; dims=2)), sum(ydot), A_obs,
                         SparseMatrixCSC{Float64,Int}(copy(transpose(A_obs))), FF, WW)
    ec = (sizes = sizes, counts = length.(class_idx),
          vtv_nzvals = vtv_nzvals, projected = projected, ydot = ydot)
    return design, K, ec, (G = G, Vg = Vg, yg = yg, class_idx = class_idx, src = src)
end

# ─── AR(1) within-firm error correlation: sigma_eps^2 R(eta), estimated ─────

"""
    build_ar1_V_stats(f_obs, w_obs, y, edge_index_obs, n_firms, n_workers,
                      match_ids=nothing)

Design-product components under an AR(1) within-firm error correlation
`Corr(eps_k, eps_l) = eta^|k-l|`: the error covariance is `sigma_eps^2 * R(eta)`
with `R(eta)` block-diagonal by firm and block `[eta^|k-l|]`, where `k, l` are
the caller-supplied `edge_index` values (a unique permutation of `1..m_i`
within each firm). The AR(1) precision is tridiagonal:

    Rinv(eta) = (S_full - eta*S_adj + eta^2*S_int) / (1 - eta^2)

with `S_full = I`, `S_adj` the within-firm edge adjacency (|k-l| == 1), and
`S_int` the diagonal correction: +1 on interior edges (excludes the first and
last edge of each firm), -1 on singleton-firm edges (whose precision is exactly
1, cancelling the `1/(1-eta^2)` base). The `R(eta)`-weighted design products
are the corresponding three-way linear combination, so this returns the three
precomputed components aligned to the **union pattern**
`V'V + V'S_adj V + V'S_int V` rather than a single design. The union is
needed because adjacent observations with distinct workers create
worker–worker entries in `V'S_adj V` that `V'V` need not contain (and the
`S_int` −1 on singleton diagonals would cancel `V'V` entries under a plain
sparse sum).

With `match_ids`, edges sharing an id form **one** observation exactly as in
[`build_match_V_stats`](@ref) (each distinct firm weighted `1/F`, each
distinct worker `1/M`), and the AR(1) process is defined **on the match
units**: `Corr(eps_s, eps_t) = eta^|rank_s - rank_t|` over the firm's matches
ranked by `edge_index`. A co-managed match is one spell with one error draw —
the error process lives on spells, not member rows, so no `G R_raw G'`
reweighting arises. Each match must span a single firm (the firm's spell
timeline is the process index; a multi-firm match is an `ArgumentError`), and
`edge_index` must agree across the rows of a match.

Returns `(pattern, vtv_full, vtv_adj, vtv_int, projected_full, projected_adj,
projected_int, ydot_full, ydot_adj, ydot_int, n_blocks, K, V, Sadj, Sint, yv,
src)`.
"""
function build_ar1_V_stats(
    f_obs::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    edge_index_obs::Vector{Int},
    n_firms::Int,
    n_workers::Int,
    match_ids::Union{Nothing,Vector{Int}}=nothing,
)
    n = n_firms + n_workers
    V, yv, src = observation_rows(f_obs, w_cols, y, match_ids, n_firms, n_workers)
    K = length(yv)

    # Per-observation firm and spell rank. Without match_ids these are the
    # inputs; with match_ids they are read per match (first-appearance order,
    # the same order observation_rows assigns) and must be internally
    # consistent: one firm per match, one edge_index per match.
    firm_of, eidx = if match_ids === nothing
        f_obs, edge_index_obs
    else
        (per_observation_value(f_obs, match_ids,
             "the firm index (the AR(1) process is defined on a firm's " *
             "matches, so each match must span a single firm)"),
         per_observation_value(edge_index_obs, match_ids, "edge_index"))
    end

    # Group observations by firm; validate edge_index is 1..m_i, and derive the
    # within-firm adjacency (S_adj) and interior diagonal (S_int).
    firm_groups = Dict{Int,Vector{Int}}()
    for s in 1:K
        push!(get!(firm_groups, firm_of[s], Int[]), s)
    end
    Sadj_I = Int[]; Sadj_J = Int[]; Sadj_V = Float64[]
    Sint_I = Int[]; Sint_J = Int[]; Sint_V = Float64[]
    for (_, rows) in firm_groups
        m = length(rows)
        es = eidx[rows]
        sort(es) == collect(1:m) || throw(ArgumentError(
            "error_eta requires edge_index to be a permutation of 1:$m within each firm; got $(sort(es)) for a firm with $m observations."))
        ord = sortperm(es)
        for pos in 1:m
            r = rows[ord[pos]]
            if pos < m
                rnext = rows[ord[pos+1]]
                push!(Sadj_I, r); push!(Sadj_J, rnext); push!(Sadj_V, 1.0)
                push!(Sadj_I, rnext); push!(Sadj_J, r); push!(Sadj_V, 1.0)
            end
            if 1 < pos < m
                push!(Sint_I, r); push!(Sint_J, r); push!(Sint_V, 1.0)
            elseif m == 1
                # Singleton firm: its AR(1) precision is exactly 1 (no 1/(1-eta^2)
                # scaling), so mark the diagonal correction -1 to cancel the base.
                push!(Sint_I, r); push!(Sint_J, r); push!(Sint_V, -1.0)
            end
        end
    end
    Sadj = sparse(Sadj_I, Sadj_J, Sadj_V, K, K)
    Sint = sparse(Sint_I, Sint_J, Sint_V, K, K)

    Pfull = SparseMatrixCSC{Float64,Int}(transpose(V) * V)
    Padj = SparseMatrixCSC{Float64,Int}(transpose(V) * Sadj * V)
    Pint = SparseMatrixCSC{Float64,Int}(transpose(V) * Sint * V)
    # Union pattern (see docstring): P_adj is not pattern-contained in P_full,
    # and Pint carries a -1 on singleton diagonals that would cancel Pfull and
    # be dropped by sparse +. Build the structural union from all-ones markers
    # so no position is lost to exact cancellation.
    mark(A) = SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), ones(Float64, nnz(A)))
    P = SparseMatrixCSC{Float64,Int}(mark(Pfull) + mark(Padj) + mark(Pint))
    vtv_full = align_nzvals(P, Pfull)
    vtv_adj = align_nzvals(P, Padj)
    vtv_int = align_nzvals(P, Pint)

    projected_full = Vector{Float64}(transpose(V) * yv)
    projected_adj = Vector{Float64}(transpose(V) * (Sadj * yv))
    projected_int = Vector{Float64}(transpose(V) * (Sint * yv))

    ydot_full = dot(yv, yv)
    ydot_adj = dot(yv, Sadj * yv)
    ydot_int = dot(yv, Sint * yv)

    return (pattern = P,
            vtv_full = vtv_full, vtv_adj = vtv_adj, vtv_int = vtv_int,
            projected_full = projected_full, projected_adj = projected_adj, projected_int = projected_int,
            ydot_full = ydot_full, ydot_adj = ydot_adj, ydot_int = ydot_int,
            n_blocks = length(firm_groups), K = K,
            V = V, Sadj = Sadj, Sint = Sint, y = yv, src = src)
end

"""
Three-component R-weighted mean-structure statistics for the AR(1) error
model: `(full, adj, int)` hold `V'X`, `X'X`, `X'y` under the identity,
`S_adj`, and `S_int` weightings respectively, so the per-evaluation products
are the same three-way combination as the design.
"""
function build_ar1_mean_stats(ar1_aux, X_rows::Matrix{Float64})
    V = ar1_aux.V
    SX = ar1_aux.Sadj * X_rows
    TX = ar1_aux.Sint * X_rows
    Sy = ar1_aux.Sadj * ar1_aux.y
    Ty = ar1_aux.Sint * ar1_aux.y
    p = size(X_rows, 2)
    return (
        full = MeanStats(Matrix{Float64}(transpose(V) * X_rows),
                         Matrix{Float64}(transpose(X_rows) * X_rows),
                         Vector{Float64}(transpose(X_rows) * ar1_aux.y), p),
        adj = MeanStats(Matrix{Float64}(transpose(V) * SX),
                        Matrix{Float64}(transpose(X_rows) * SX),
                        Vector{Float64}(transpose(X_rows) * Sy), p),
        int = MeanStats(Matrix{Float64}(transpose(V) * TX),
                        Matrix{Float64}(transpose(X_rows) * TX),
                        Vector{Float64}(transpose(X_rows) * Ty), p),
    )
end

"""
Reassemble the R(eta)-weighted observation products for a given `eta`. The
likelihood constant `log det R = (K - n_blocks) * log(1 - eta^2)` rides in
`WeightStats.log_weight_sum` as `-logdet R` (matching the `error_cov`
convention), so every downstream solver and the profiled mean structure work
unchanged. Returns `(design, weights, mean_stats)`.
"""
function ar1_observation_stats(ar::ErrorAR1Stats, eta::Float64, n_firms::Int)
    denom = 1.0 - eta^2
    denom > 0 || throw(ArgumentError("eta must satisfy -1 < eta < 1; got $(eta)."))
    c_full = 1.0 / denom
    c_adj = -eta / denom
    c_int = eta^2 / denom
    nzvals = c_full .* ar.vtv_full .+ c_adj .* ar.vtv_adj .+ c_int .* ar.vtv_int
    n = size(ar.pattern, 1)
    VtV = SparseMatrixCSC(n, n, ar.pattern.colptr, ar.pattern.rowval, nzvals)
    FF = SparseMatrixCSC{Float64,Int}(VtV[1:n_firms, 1:n_firms])
    WW = SparseMatrixCSC{Float64,Int}(VtV[(n_firms+1):n, (n_firms+1):n])
    A_obs = SparseMatrixCSC{Float64,Int}(VtV[1:n_firms, (n_firms+1):n])
    design = DesignStats(
        VtV,
        c_full .* ar.projected_full .+ c_adj .* ar.projected_adj .+ c_int .* ar.projected_int,
        c_full * ar.ydot_full + c_adj * ar.ydot_adj + c_int * ar.ydot_int,
        A_obs,
        SparseMatrixCSC{Float64,Int}(copy(transpose(A_obs))),
        FF,
        WW,
    )
    logdet_R = (ar.K - ar.n_blocks) * log(denom)
    weights = WeightStats(-logdet_R, Float64(ar.K), Float64(ar.K), 1.0, 1.0)
    ms = ar.mean_full === nothing ? nothing :
        MeanStats(c_full .* ar.mean_full.VtX .+ c_adj .* ar.mean_adj.VtX .+ c_int .* ar.mean_int.VtX,
                  c_full .* ar.mean_full.XtX .+ c_adj .* ar.mean_adj.XtX .+ c_int .* ar.mean_int.XtX,
                  c_full .* ar.mean_full.Xty .+ c_adj .* ar.mean_adj.Xty .+ c_int .* ar.mean_int.Xty,
                  ar.mean_full.p)
    return design, weights, ms
end

# ─── Per-firm error blocks: edge-level, never collapsed ────────────────────

"""
    build_block_V_stats(f_obs, w_obs, y_obs, firm_group, n_firms, n_workers,
                        match_ids=nothing)

Design products for `error_blocks=:iw`: each firm's observations form one
error block modeled by the active block solver ([`EMIWBlocks`](@ref), which
integrates the block covariance out). Blocks are discovered from `firm_group`
(first-appearance order), never parametrized.

Without `match_ids` the observations are the input rows. With `match_ids`,
edges sharing an id form **one** observation exactly as in
[`build_match_V_stats`](@ref) (each distinct firm weighted `1/F`, each
distinct worker `1/M`), and the firm's block is its **matches** — `Ψ_{m_i}`
acts on match errors, `m_i` = matches at firm `i`. Each match must span a
single firm (a multi-firm match is an `ArgumentError`).

Returns `(V, yv, block_of, sizes)`. Each `firm_group` value must span a single
firm (the grouping key is the firm id per row).
"""
function build_block_V_stats(
    f_obs::Vector{Int},
    w_obs::Vector{Int},
    y_obs::Vector{Float64},
    firm_group::Vector{Int},
    n_firms::Int,
    n_workers::Int,
    match_ids::Union{Nothing,Vector{Int}}=nothing,
)
    V, yv, _ = observation_rows(f_obs, w_obs, y_obs, match_ids, n_firms, n_workers)
    K = length(yv)

    # firm_group must be the rowwise firm mapping: the estimator has one latent
    # mixing weight u_i per firm, so a firm split across several labels (or a
    # label spanning firms) would fit a different model than the API claims.
    firm_group == f_obs || throw(ArgumentError(
        "error_blocks=:iw requires firm_group to equal the firm id per row; " *
        "each row's firm_group value must match its firm index."))

    firm_of = match_ids === nothing ? f_obs :
        per_observation_value(f_obs, match_ids,
            "the firm index (an error block is one firm's matches, so each " *
            "match must span a single firm)")

    gpos = Dict{Int,Int}()
    members = Vector{Vector{Int}}()
    for s in 1:K
        g = get!(gpos, firm_of[s]) do
            push!(members, Int[])
            length(members)
        end
        push!(members[g], s)
    end
    B = length(members)
    block_of = zeros(Int, K)
    sizes = Vector{Int}(undef, B)
    for i in 1:B
        rows = members[i]
        sizes[i] = length(rows)
        for s in rows
            block_of[s] = i
        end
    end
    return V, yv, block_of, sizes
end
