# ═══════════════════════════════════════════════════════════════════════════
# suffstats: compute sufficient statistics from raw data
# ═══════════════════════════════════════════════════════════════════════════

"""
    suffstats(::Type{<:AbstractBipartiteModel}, df::DataFrame;
              outcome=:y, firm_id=:firm_id, worker_id=:worker_id,
              weighting=Weighting(), model_adjacency=:binary,
              max_degree=nothing, standardize=true, on_missing=:drop)

Compute sufficient statistics for bipartite GMRF MLE from a DataFrame.

Returns a [`BipartiteGMRFStats`](@ref) that can be passed to [`fit_mle`](@ref).
Extends `Distributions.suffstats`.

`max_degree` filters on the number of *observation rows* per firm and per
worker (not the number of distinct partners); entities with more rows than
`max_degree` are dropped together with their rows.
"""
function suffstats(
    ::Type{M},
    df::DataFrame;
    outcome::Symbol=:y,
    firm_id::Symbol=:firm_id,
    worker_id::Symbol=:worker_id,
    weighting::Weighting=Weighting(),
    model_adjacency::Symbol=:binary,
    max_degree::Union{Nothing,Int}=nothing,
    standardize::Bool=true,
    on_missing::Symbol=:drop,
) where {M<:AbstractBipartiteModel}
    on_missing in (:drop, :error) ||
        throw(ArgumentError("on_missing must be :drop or :error; got $(on_missing)."))
    model_adjacency in (:binary, :counts) ||
        throw(ArgumentError("model_adjacency must be :binary or :counts; got $(model_adjacency)."))

    for c in (outcome, firm_id, worker_id)
        hasproperty(df, c) || throw(ArgumentError("Missing required column $(c)."))
    end
    nrow(df) > 0 || throw(ArgumentError("Empty dataset."))

    M <: BipartiteVarianceStableModel && weighting.observations != :raw &&
        throw(ArgumentError("BipartiteVarianceStableModel currently supports only Weighting(observations=:raw)."))

    d0 = max_degree === nothing ? df : filter_max_degree(df, max_degree, firm_id, worker_id)
    nrow(d0) > 0 || throw(ArgumentError("No rows remain after max_degree filtering."))

    y_raw_all = map(to_float_nan, d0[!, outcome])
    keep = map(isfinite, y_raw_all)
    if on_missing == :error && !all(keep)
        throw(ArgumentError("Outcome $(outcome) contains missing or non-finite values."))
    end
    d = d0[keep, :]
    y_raw = Float64.(y_raw_all[keep])
    personyear_rows = length(y_raw)
    personyear_rows > 0 || throw(ArgumentError("No usable observations after missing filter."))

    y_mean = standardize ? mean(y_raw) : 0.0
    y_std = standardize ? std(y_raw) : 1.0
    isfinite(y_std) && y_std > 0 || throw(ArgumentError("Outcome has zero or invalid standard deviation."))

    work = DataFrame(
        firm = d[!, firm_id],
        worker = d[!, worker_id],
        y = y_raw,
    )
    collapsed_all = combine(groupby(work, [:firm, :worker]),
        :y => length => :T,
        :y => mean => :y_mean,
        :y => edge_ssw => :ssw_raw,
    )

    local ids_f, ids_w, n_firms, n_workers, k
    local f_rows, w_cols, y, T_edge
    local base_stats, A_prior_base, decomp_f_rows, decomp_w_cols, decomp_y, decomp_T
    local within_ss, within_df, personyear_within_ss, rho_eps_likelihood
    local log_weight_sum, effective_weight_sum, effective_weight_over_T_sum
    local mean_effective_weight, max_effective_weight
    local firm_to_index, worker_to_index
    obs = weighting.observations

    if obs == :raw
        ids_f = collect(unique(work.firm))
        ids_w = collect(unique(work.worker))
        n_firms = length(ids_f)
        n_workers = length(ids_w)
        k = nrow(work)
        firm_to_index = Dict(id => i for (i, id) in enumerate(ids_f))
        worker_to_index = Dict(id => i for (i, id) in enumerate(ids_w))
        f_rows = Vector{Int}(undef, k)
        w_cols = Vector{Int}(undef, k)
        @inbounds for i in 1:k
            f_rows[i] = firm_to_index[work.firm[i]]
            w_cols[i] = worker_to_index[work.worker[i]]
        end
        y = (Float64.(work.y) .- y_mean) ./ y_std
        T_edge = ones(Int, k)
        base_stats = build_V_stats(f_rows, w_cols, y, n_firms, n_workers)
        A_prior_base = copy(base_stats.A_obs)

        decomp_f_rows = Vector{Int}(undef, nrow(collapsed_all))
        decomp_w_cols = Vector{Int}(undef, nrow(collapsed_all))
        @inbounds for i in 1:nrow(collapsed_all)
            decomp_f_rows[i] = firm_to_index[collapsed_all.firm[i]]
            decomp_w_cols[i] = worker_to_index[collapsed_all.worker[i]]
        end
        decomp_y = (Float64.(collapsed_all.y_mean) .- y_mean) ./ y_std
        decomp_T = Int.(collapsed_all.T)
        personyear_within_ss = sum(Float64.(collapsed_all.ssw_raw)) / y_std^2
        within_ss = 0.0
        within_df = 0
        rho_eps_likelihood = nothing
        log_weight_sum = 0.0
        effective_weight_sum = Float64(k)
        effective_weight_over_T_sum = Float64(k)
        mean_effective_weight = 1.0
        max_effective_weight = 1.0
    else
        ids_f = collect(unique(collapsed_all.firm))
        ids_w = collect(unique(collapsed_all.worker))
        n_firms = length(ids_f)
        n_workers = length(ids_w)
        k = nrow(collapsed_all)
        firm_to_index = Dict(id => i for (i, id) in enumerate(ids_f))
        worker_to_index = Dict(id => i for (i, id) in enumerate(ids_w))
        f_rows = Vector{Int}(undef, k)
        w_cols = Vector{Int}(undef, k)
        @inbounds for i in 1:k
            f_rows[i] = firm_to_index[collapsed_all.firm[i]]
            w_cols[i] = worker_to_index[collapsed_all.worker[i]]
        end
        y = (Float64.(collapsed_all.y_mean) .- y_mean) ./ y_std
        T_edge = Int.(collapsed_all.T)
        A_prior_base = sparse(f_rows, w_cols, Float64.(T_edge), n_firms, n_workers)
        decomp_f_rows = copy(f_rows)
        decomp_w_cols = copy(w_cols)
        decomp_y = copy(y)
        decomp_T = copy(T_edge)
        personyear_within_ss = sum(Float64.(collapsed_all.ssw_raw)) / y_std^2

        if obs == :edge
            base_stats = build_weighted_V_stats(f_rows, w_cols, y, ones(Float64, k), n_firms, n_workers)
            within_ss = 0.0
            within_df = 0
            rho_eps_likelihood = nothing
            log_weight_sum = 0.0
            effective_weight_sum = Float64(k)
            effective_weight_over_T_sum = Float64(k)
            mean_effective_weight = 1.0
            max_effective_weight = 1.0
        else
            rho0 = weighting.rho_eps == :estimate ? 0.5 : Float64(weighting.rho_eps)
            base_stats = build_match_weight_stats(f_rows, w_cols, y, T_edge, n_firms, n_workers, rho0)
            within_ss = personyear_within_ss
            within_df = personyear_rows - k
            rho_eps_likelihood = rho0
            log_weight_sum = base_stats.log_weight_sum
            effective_weight_sum = base_stats.effective_weight_sum
            effective_weight_over_T_sum = base_stats.effective_weight_over_T_sum
            mean_effective_weight = base_stats.mean_effective_weight
            max_effective_weight = base_stats.max_effective_weight
        end
    end

    A_prior = copy(A_prior_base)
    model_adjacency == :binary && (A_prior.nzval .= 1.0)

    unique_edges = nnz(sparse(f_rows, w_cols, ones(Float64, length(f_rows)), n_firms, n_workers))
    duplicate_rows = personyear_rows - unique_edges
    mean_edge_count = unique_edges > 0 ? Float64(personyear_rows) / Float64(unique_edges) : NaN
    max_edge_count = isempty(T_edge) ? NaN : Float64(maximum(T_edge))

    metadata = (
        outcome = outcome,
        firm_id = firm_id,
        worker_id = worker_id,
        max_degree = max_degree,
        model_adjacency = model_adjacency,
        unique_edges = unique_edges,
        duplicate_rows = duplicate_rows,
        mean_edge_count = mean_edge_count,
        max_edge_count = max_edge_count,
        total_prior_weight = sum(A_prior.nzval),
        max_prior_degree_f = maximum(vec(sum(A_prior; dims=2))),
        max_prior_degree_w = maximum(vec(sum(A_prior; dims=1))),
    )

    return BipartiteGMRFStats(
        base_stats.VtV,
        base_stats.projected_y,
        Float64(base_stats.ydot),
        base_stats.A_obs,
        base_stats.At_obs,
        base_stats.cnt_f,
        base_stats.cnt_w,
        A_prior,
        f_rows,
        w_cols,
        y,
        T_edge,
        decomp_f_rows,
        decomp_w_cols,
        decomp_y,
        decomp_T,
        ids_f,
        ids_w,
        firm_to_index,
        worker_to_index,
        n_firms,
        n_workers,
        k,
        personyear_rows,
        Float64(y_mean),
        Float64(y_std),
        standardize,
        weighting,
        rho_eps_likelihood,
        Float64(within_ss),
        Int(within_df),
        Float64(personyear_within_ss),
        Float64(log_weight_sum),
        Float64(effective_weight_sum),
        Float64(effective_weight_over_T_sum),
        Float64(mean_effective_weight),
        Float64(max_effective_weight),
        metadata,
    )
end
