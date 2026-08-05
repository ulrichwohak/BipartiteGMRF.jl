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
    VtV = [spdiagm(0 => cnt_f) A_obs; copy(transpose(A_obs)) spdiagm(0 => cnt_w)]
    return DesignStats(
        VtV,
        vcat(proj_f, proj_w),
        dot(y, y),
        A_obs,
        copy(transpose(A_obs)),
        cnt_f,
        cnt_w,
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
    VtV = [spdiagm(0 => cnt_f) A_obs; copy(transpose(A_obs)) spdiagm(0 => cnt_w)]
    return DesignStats(
        VtV,
        vcat(proj_f, proj_w),
        ydot,
        A_obs,
        copy(transpose(A_obs)),
        cnt_f,
        cnt_w,
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
