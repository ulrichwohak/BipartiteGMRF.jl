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

# ─── Banded within-firm error correlation (issue: eta) ───────────────────────

"""
    build_banded_V_stats(f_obs, w_obs, y, match_ids, rank_obs, band, n_firms, n_workers)

Design products under a banded within-firm error correlation: for two
observations of the SAME firm at spell ranks `r` and `r'`,
`Corr(eps, eps') = band[|r - r'|]` (1 on the diagonal, 0 beyond the band), and
zero across firms. `R` is block-diagonal by firm, so its inverse is a set of
small dense blocks; each block must be PD, and the error names the block size
that fails.

Returns `(design, logdet_R, K, aux)`. `design` carries `V'R⁻¹V`, `V'R⁻¹y` and
`y'R⁻¹y` in the usual `DesignStats` fields, so every downstream consumer —
`M = Q + λV'V`, the quadratic form, the Hutchinson blocks — works unchanged;
`logdet_R` enters the likelihood through `WeightStats.log_weight_sum` (as
`-logdet_R`, matching the sign convention `2·NLL ⊃ -log_weight_sum`). `aux`
holds `(V, Rinv, y, src)` so mean-structure products can be R-weighted without
rebuilding; `src` maps each observation to its first input row.

With `match_ids`, edges sharing an id form one observation exactly as in
`build_match_V_stats`, except that a banded match must sit at a SINGLE firm
(the band is a within-firm object) and its rank must be constant within the
match.
"""
function build_banded_V_stats(
    f_obs::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    match_ids::Union{Nothing,Vector{Int}},
    rank_obs::Vector{Int},
    band::Vector{Float64},
    n_firms::Int,
    n_workers::Int,
)
    n = n_firms + n_workers
    Vi = Int[]; Vj = Int[]; Vv = Float64[]
    yv = Float64[]; firm_of = Int[]; rank_of = Int[]; src = Int[]
    if match_ids === nothing
        for i in eachindex(y)
            push!(yv, y[i]); push!(firm_of, f_obs[i]); push!(rank_of, rank_obs[i]); push!(src, i)
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
                push!(yv, y[i]); push!(rank_of, rank_obs[i])
                push!(firm_of, f_obs[i]); push!(src, i)
                length(yv)
            end
            rank_obs[i] == rank_of[s] || throw(ArgumentError(
                "match $(match_ids[i]) has an inconsistent spell rank."))
            push!(firms_of[s], f_obs[i])
            push!(workers_of[s], w_cols[i])
        end
        for s in eachindex(yv)
            fs = unique(firms_of[s]); ws = unique(workers_of[s])
            length(fs) == 1 || throw(ArgumentError(
                "error bands require single-firm matches; a match spans $(length(fs)) firms."))
            push!(Vi, s); push!(Vj, fs[1]); push!(Vv, 1.0)
            for w in ws
                push!(Vi, s); push!(Vj, n_firms + w); push!(Vv, 1.0 / length(ws))
            end
        end
    end
    K = length(yv)
    V = sparse(Vi, Vj, Vv, K, n)

    byfirm = Dict{Int,Vector{Int}}()
    for i in 1:K
        push!(get!(byfirm, firm_of[i], Int[]), i)
    end
    Ri = Int[]; Rj = Int[]; Rv = Float64[]
    logdet_R = 0.0
    for (_, idx) in byfirm
        kf = length(idx)
        if kf == 1
            push!(Ri, idx[1]); push!(Rj, idx[1]); push!(Rv, 1.0)
            continue
        end
        Rf = Matrix{Float64}(I, kf, kf)
        for a in 1:kf, b in (a+1):kf
            d = abs(rank_of[idx[a]] - rank_of[idx[b]])
            1 <= d <= length(band) && (Rf[a, b] = Rf[b, a] = band[d])
        end
        C = cholesky(Symmetric(Rf); check=false)
        issuccess(C) || throw(ArgumentError(
            "banded error correlation is not positive definite for a firm with $(kf) observations; shrink the band."))
        logdet_R += logdet(C)
        Rinv_f = inv(C)
        for a in 1:kf, b in 1:kf
            push!(Ri, idx[a]); push!(Rj, idx[b]); push!(Rv, Rinv_f[a, b])
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
R-weighted mean-structure products for the banded error model. Storing
`V'R⁻¹X`, `X'R⁻¹X` and `X'R⁻¹y` makes `mean_profile_correction` correct
without modification: its `c` and `G` become the GLS versions automatically.
"""
function build_banded_mean_stats(aux, X_rows::Matrix{Float64})
    RX = aux.Rinv * X_rows
    return MeanStats(
        Matrix{Float64}(transpose(aux.V) * RX),
        Matrix{Float64}(transpose(X_rows) * RX),
        Vector{Float64}(transpose(X_rows) * (aux.Rinv * aux.y)),
        size(X_rows, 2),
    )
end
