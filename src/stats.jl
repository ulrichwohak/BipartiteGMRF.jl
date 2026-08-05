# ═══════════════════════════════════════════════════════════════════════════
# suffstats: compute sufficient statistics from raw data
# ═══════════════════════════════════════════════════════════════════════════

"""
    suffstats(::Type{<:AbstractBipartiteModel}, f_idx, w_idx, y;
              n_firms=maximum(f_idx), n_workers=maximum(w_idx),
              weighting=Weighting(), model_adjacency=:binary,
              match_id=nothing, standardize=true)

Compute sufficient statistics for bipartite GMRF MLE from parallel observation
vectors: `f_idx[k]` and `w_idx[k]` are the 1-based firm and worker indices of
observation `k`, and `y[k]` its outcome. Repeated `(firm, worker)` pairs are
allowed and handled according to `weighting`.

Returns a [`BipartiteGMRFStats`](@ref) that can be passed to [`fit_mle`](@ref).
Extends `Distributions.suffstats`.

## Graph-only edges

Non-finite values in `y` (NaN, ±Inf) mark *graph-only* edges: the
corresponding `(f_idx[k], w_idx[k])` pair enters the prior adjacency
`A_prior` (and therefore the model's precision matrix and node degrees)
but contributes no design row, no term to `V'y`, `y'y`, or the
observation count `K`. At least one finite outcome is required.

## Match-grouped observations

When `match_id` is provided, edges sharing a match id form **one**
observation whose design row averages `1/F_s` over each distinct firm
and `1/M_s` over each distinct worker in the match. Outcomes within a
match must agree (up to machine precision). `K` counts matches, not
edges. Currently requires `Weighting(observations=:raw)`.
"""
function suffstats(
    ::Type{M},
    f_idx::AbstractVector{<:Integer},
    w_idx::AbstractVector{<:Integer},
    y::AbstractVector{<:Real};
    n_firms::Integer=isempty(f_idx) ? 0 : maximum(f_idx),
    n_workers::Integer=isempty(w_idx) ? 0 : maximum(w_idx),
    weighting::Weighting=Weighting(),
    model_adjacency::Symbol=:binary,
    match_id::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    standardize::Bool=true,
) where {M<:AbstractBipartiteModel}
    model_adjacency in (:binary, :counts) ||
        throw(ArgumentError("model_adjacency must be :binary or :counts; got $(model_adjacency)."))
    M <: BipartiteVarianceStableModel && weighting.observations != :raw &&
        throw(ArgumentError("BipartiteVarianceStableModel currently supports only Weighting(observations=:raw)."))
    match_id !== nothing && weighting.observations != :raw &&
        throw(ArgumentError("match_id grouping currently supports only Weighting(observations=:raw)."))

    length(y) > 0 || throw(ArgumentError("Empty dataset."))
    length(f_idx) == length(y) && length(w_idx) == length(y) ||
        throw(ArgumentError("f_idx, w_idx, and y must have the same length."))
    match_id !== nothing && length(match_id) != length(y) &&
        throw(ArgumentError("match_id must have the same length as y."))
    n_f = Int(n_firms)
    n_w = Int(n_workers)
    for k in eachindex(f_idx)
        1 <= f_idx[k] <= n_f || throw(ArgumentError("firm index $(f_idx[k]) outside 1:$(n_f)."))
        1 <= w_idx[k] <= n_w || throw(ArgumentError("worker index $(w_idx[k]) outside 1:$(n_w)."))
    end

    y_raw = Float64.(y)
    f_rows = Int.(f_idx)
    w_cols = Int.(w_idx)

    # Separate observed (finite outcome) from graph-only (NaN/Inf) edges
    obs_mask = isfinite.(y_raw)
    n_obs = count(obs_mask)
    n_obs > 0 || throw(ArgumentError("No finite outcomes in y."))

    y_finite = y_raw[obs_mask]

    # ── match_id validation (before standardization) ──
    match_id_obs = nothing
    match_outcomes = nothing
    if match_id !== nothing
        mid_all = Int.(match_id)
        # No match may mix finite and NaN outcomes
        seen_finite = Dict{Int,Bool}()
        for k in eachindex(mid_all)
            is_fin = obs_mask[k]
            prev = get(seen_finite, mid_all[k], nothing)
            if prev === nothing
                seen_finite[mid_all[k]] = is_fin
            elseif prev != is_fin
                throw(ArgumentError(
                    "Match $(mid_all[k]) mixes finite and non-finite outcomes.",
                ))
            end
        end
        # Outcomes within a match must be constant (checked on raw values)
        match_id_obs = mid_all[obs_mask]
        match_outcome = Dict{Int,Float64}()
        tol = sqrt(eps(Float64))
        for k in eachindex(match_id_obs)
            mid = match_id_obs[k]
            prev = get(match_outcome, mid, nothing)
            if prev === nothing
                match_outcome[mid] = y_finite[k]
            elseif abs(prev - y_finite[k]) > tol * (1.0 + abs(prev))
                throw(ArgumentError(
                    "Match $(mid) has inconsistent outcomes.",
                ))
            end
        end
        match_outcomes = collect(values(match_outcome))
    end

    # Standardize: match-weighted when match_id is provided
    if standardize
        if match_outcomes !== nothing
            y_mean = mean(match_outcomes)
            y_std = std(match_outcomes)
        else
            y_mean = mean(y_finite)
            y_std = std(y_finite)
        end
    else
        y_mean = 0.0
        y_std = 1.0
    end
    isfinite(y_std) && y_std > 0 || throw(ArgumentError("Outcome has zero or invalid standard deviation."))

    f_obs = f_rows[obs_mask]
    w_obs = w_cols[obs_mask]
    y_obs_scaled = (y_finite .- y_mean) ./ y_std
    personyear_rows = n_obs

    edges = collapse_edges(f_obs, w_obs, y_obs_scaled)
    n_edges = length(edges.f)
    decomp = EdgeData(edges.f, edges.w, edges.y_mean, edges.T)
    personyear_within_ss = sum(edges.ssw)

    obs = weighting.observations
    block = if obs == :raw
        design = if match_id_obs !== nothing
            build_match_V_stats(f_obs, w_obs, y_obs_scaled, match_id_obs, n_f, n_w)
        else
            build_V_stats(f_obs, w_obs, y_obs_scaled, n_f, n_w)
        end
        n_matches = match_id_obs !== nothing ? length(unique(match_id_obs)) : personyear_rows
        (
            K = n_matches,
            base = EdgeData(f_obs, w_obs, y_obs_scaled, ones(Int, personyear_rows)),
            design = design,
            weights = trivial_weight_stats(n_matches),
            within_ss = 0.0,
            within_df = 0,
            rho_eps_likelihood = nothing,
        )
    elseif obs == :edge
        design = build_weighted_V_stats(edges.f, edges.w, edges.y_mean, ones(Float64, n_edges), n_f, n_w)
        (
            K = n_edges,
            base = EdgeData(edges.f, edges.w, edges.y_mean, edges.T),
            design = design,
            weights = trivial_weight_stats(n_edges),
            within_ss = 0.0,
            within_df = 0,
            rho_eps_likelihood = nothing,
        )
    else  # :effective
        rho0 = weighting.rho_eps
        design, weight_stats = build_match_weight_stats(edges.f, edges.w, edges.y_mean, edges.T, n_f, n_w, rho0)
        (
            K = n_edges,
            base = EdgeData(edges.f, edges.w, edges.y_mean, edges.T),
            design = design,
            weights = weight_stats,
            within_ss = personyear_within_ss,
            within_df = personyear_rows - n_edges,
            rho_eps_likelihood = rho0,
        )
    end

    # A_prior from ALL edges (including graph-only); design uses observed only
    A_prior = sparse(f_rows, w_cols, ones(Float64, length(f_rows)), n_f, n_w)
    model_adjacency == :binary && (A_prior.nzval .= 1.0)

    n_graph_only = length(f_rows) - n_obs
    duplicate_rows = personyear_rows - n_edges
    metadata = (
        model_adjacency = model_adjacency,
        unique_edges = n_edges,
        duplicate_rows = duplicate_rows,
        mean_edge_count = Float64(personyear_rows) / Float64(n_edges),
        max_edge_count = Float64(maximum(edges.T)),
        total_prior_weight = sum(A_prior.nzval),
        max_prior_degree_f = maximum(vec(sum(A_prior; dims=2))),
        max_prior_degree_w = maximum(vec(sum(A_prior; dims=1))),
        graph_only_rows = n_graph_only,
        graph_only_edges = nnz(A_prior) - n_edges,
    )

    return BipartiteGMRFStats(
        block.design,
        A_prior,
        block.base,
        decomp,
        n_f,
        n_w,
        block.K,
        personyear_rows,
        y_mean,
        y_std,
        standardize,
        weighting,
        block.rho_eps_likelihood,
        block.within_ss,
        block.within_df,
        personyear_within_ss,
        block.weights,
        metadata,
    )
end
