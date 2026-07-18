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
    return (
        projected_y = vcat(proj_f, proj_w),
        VtV = VtV,
        cnt_f = cnt_f,
        cnt_w = cnt_w,
        A_obs = A_obs,
        At_obs = copy(transpose(A_obs)),
        ydot = dot(y, y),
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
    return (
        projected_y = vcat(proj_f, proj_w),
        VtV = VtV,
        cnt_f = cnt_f,
        cnt_w = cnt_w,
        A_obs = A_obs,
        At_obs = copy(transpose(A_obs)),
        ydot = ydot,
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
    stats = build_weighted_V_stats(f_rows, w_cols, y, weights, n_firms, n_workers)
    return merge(stats, (
        log_weight_sum = sum(log, weights),
        effective_weight_sum = sum(weights),
        mean_effective_weight = mean(weights),
        max_effective_weight = maximum(weights),
        effective_weight_over_T_sum = sum(weights ./ Float64.(T)),
    ))
end

function filter_max_degree(df::DataFrame, max_degree::Int, firm_id::Symbol, worker_id::Symbol)
    max_degree > 0 || throw(ArgumentError("max_degree must be positive."))
    firm_deg = combine(groupby(df, firm_id), nrow => :deg)
    worker_deg = combine(groupby(df, worker_id), nrow => :deg)
    keep_firms = Set(firm_deg[firm_deg.deg .<= max_degree, firm_id])
    keep_workers = Set(worker_deg[worker_deg.deg .<= max_degree, worker_id])
    return filter(row -> row[firm_id] in keep_firms && row[worker_id] in keep_workers, df)
end

function with_observation_stats(gmrf_stats::BipartiteGMRFStats, obs_stats, rho_eps::Union{Nothing,Float64})
    return BipartiteGMRFStats(
        obs_stats.VtV,
        obs_stats.projected_y,
        Float64(obs_stats.ydot),
        obs_stats.A_obs,
        obs_stats.At_obs,
        obs_stats.cnt_f,
        obs_stats.cnt_w,
        gmrf_stats.A_prior,
        gmrf_stats.base_f_rows,
        gmrf_stats.base_w_cols,
        gmrf_stats.base_y,
        gmrf_stats.base_T,
        gmrf_stats.decomp_f_rows,
        gmrf_stats.decomp_w_cols,
        gmrf_stats.decomp_y,
        gmrf_stats.decomp_T,
        gmrf_stats.firm_ids,
        gmrf_stats.worker_ids,
        gmrf_stats.firm_to_index,
        gmrf_stats.worker_to_index,
        gmrf_stats.N_firms,
        gmrf_stats.N_workers,
        gmrf_stats.K,
        gmrf_stats.personyear_rows,
        gmrf_stats.y_mean,
        gmrf_stats.y_std,
        gmrf_stats.standardize,
        gmrf_stats.weighting,
        rho_eps,
        gmrf_stats.within_ss,
        gmrf_stats.within_df,
        gmrf_stats.personyear_within_ss,
        Float64(get(obs_stats, :log_weight_sum, gmrf_stats.log_weight_sum)),
        Float64(get(obs_stats, :effective_weight_sum, gmrf_stats.effective_weight_sum)),
        Float64(get(obs_stats, :effective_weight_over_T_sum, gmrf_stats.effective_weight_over_T_sum)),
        Float64(get(obs_stats, :mean_effective_weight, gmrf_stats.mean_effective_weight)),
        Float64(get(obs_stats, :max_effective_weight, gmrf_stats.max_effective_weight)),
        gmrf_stats.metadata,
    )
end
