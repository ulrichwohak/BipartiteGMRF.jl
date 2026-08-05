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

The estimator does not manipulate data: every index must be in range, every
node index in `1:n_firms` (resp. `1:n_workers`) must appear at least once,
and `y` must be free of missing or non-finite values. Map entity identifiers
to dense integer indices and drop unusable rows before calling.
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

    personyear_rows = length(y)
    personyear_rows > 0 || throw(ArgumentError("Empty dataset."))
    length(f_idx) == personyear_rows && length(w_idx) == personyear_rows ||
        throw(ArgumentError("f_idx, w_idx, and y must have the same length."))
    all(isfinite, y) || throw(ArgumentError(
        "y contains missing or non-finite values; filter the data before calling suffstats.",
    ))
    n_f = Int(n_firms)
    n_w = Int(n_workers)
    for k in eachindex(f_idx)
        1 <= f_idx[k] <= n_f || throw(ArgumentError("firm index $(f_idx[k]) outside 1:$(n_f)."))
        1 <= w_idx[k] <= n_w || throw(ArgumentError("worker index $(w_idx[k]) outside 1:$(n_w)."))
    end

    y_raw = Float64.(y)
    y_mean = standardize ? mean(y_raw) : 0.0
    y_std = standardize ? std(y_raw) : 1.0
    isfinite(y_std) && y_std > 0 || throw(ArgumentError("Outcome has zero or invalid standard deviation."))

    f_rows = Int.(f_idx)
    w_cols = Int.(w_idx)
    y_scaled = (y_raw .- y_mean) ./ y_std

    edges = collapse_edges(f_rows, w_cols, y_scaled)
    n_edges = length(edges.f)
    decomp = EdgeData(edges.f, edges.w, edges.y_mean, edges.T)
    personyear_within_ss = sum(edges.ssw)

    obs = weighting.observations
    block = if obs == :raw
        design = build_V_stats(f_rows, w_cols, y_scaled, n_f, n_w)
        (
            K = personyear_rows,
            base = EdgeData(f_rows, w_cols, y_scaled, ones(Int, personyear_rows)),
            design = design,
            weights = trivial_weight_stats(personyear_rows),
            within_ss = 0.0,
            within_df = 0,
            rho_eps_likelihood = nothing,
            A_prior_base = copy(design.A_obs),   # duplicate entries sum to counts
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
            A_prior_base = sparse(edges.f, edges.w, Float64.(edges.T), n_f, n_w),
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
            A_prior_base = sparse(edges.f, edges.w, Float64.(edges.T), n_f, n_w),
        )
    end

    A_prior = copy(block.A_prior_base)
    model_adjacency == :binary && (A_prior.nzval .= 1.0)

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
