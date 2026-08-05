# ═══════════════════════════════════════════════════════════════════════════
# suffstats: compute sufficient statistics from raw data
# ═══════════════════════════════════════════════════════════════════════════

"""
    suffstats(::Type{<:AbstractBipartiteModel}, f_idx, w_idx, y;
              n_firms=maximum(f_idx), n_workers=maximum(w_idx),
              weighting=Weighting(), model_adjacency=:binary,
              standardize=true)

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
    standardize::Bool=true,
) where {M<:AbstractBipartiteModel}
    model_adjacency in (:binary, :counts) ||
        throw(ArgumentError("model_adjacency must be :binary or :counts; got $(model_adjacency)."))
    M <: BipartiteVarianceStableModel && weighting.observations != :raw &&
        throw(ArgumentError("BipartiteVarianceStableModel currently supports only Weighting(observations=:raw)."))

    length(y) > 0 || throw(ArgumentError("Empty dataset."))
    length(f_idx) == length(y) && length(w_idx) == length(y) ||
        throw(ArgumentError("f_idx, w_idx, and y must have the same length."))
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
    y_mean = standardize ? mean(y_finite) : 0.0
    y_std = standardize ? std(y_finite) : 1.0
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
        design = build_V_stats(f_obs, w_obs, y_obs_scaled, n_f, n_w)
        (
            K = personyear_rows,
            base = EdgeData(f_obs, w_obs, y_obs_scaled, ones(Int, personyear_rows)),
            design = design,
            weights = trivial_weight_stats(personyear_rows),
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
        graph_only_edges = nnz(A_prior) - nnz(block.design.A_obs),
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
