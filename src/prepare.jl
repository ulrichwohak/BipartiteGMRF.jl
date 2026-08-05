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
