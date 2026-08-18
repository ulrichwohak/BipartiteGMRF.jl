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

## Correlated errors

`error_cov = R` replaces the i.i.d. error covariance `σ_ε² I` with
`σ_ε² R`: a sparse symmetric matrix over input rows whose connected
blocks (read off the sparsity pattern) must each be positive definite —
beyond that the blocks are arbitrary, any correlation pattern and any
within-block heteroskedasticity. The overall scale of `R` is not a free
parameter: it is rescaled internally to `tr(R) = K`, so `σ_ε²` keeps a
fixed meaning as the mean error variance regardless of the scale passed
in. The typical use is block-diagonal by firm (nonzero entries only
between same-firm observations), but the structure is up to the caller.
All sufficient statistics become `R`-weighted (`V'R⁻¹V`, `V'R⁻¹y`,
`y'R⁻¹y`) and the constant `log det R` is carried in the weight
statistics, so every downstream solver and the profiled mean structure
work unchanged. Requires `Weighting(observations=:raw)`; composes with
`match_id` (`R` is read at each match's first row, and entries between
rows of one match are dropped with the duplicate rows).

## Group-robust errors

`error_groups = g` (one group id per input row, typically the firm) makes
the fit robust to an **arbitrary unknown** PD error covariance within each
group, in the spirit of clustered standard errors. Observations sharing a
group id are collapsed to their group mean (design rows averaged, outcomes
averaged), so the unknown within-group error covariance enters the
likelihood only through the scalar variance of the group mean. That scalar
is `σ_ε² ω_c`, one free `ω` per group-size class, estimated jointly with
the structural parameters; groups of size `error_group_cap` (default 8) or
larger share the top class. The smallest class present is pinned at
`ω = 1`, which sets the scale of `σ_ε` (with singleton groups present,
`σ_ε` keeps its usual meaning exactly). Within-group contrasts carry no
information under this model — by design: their error covariance is
unknown. Fitted `ω` are reported in the result metadata
(`error_class_variances`, `error_class_sizes`). Requires
`Weighting(observations=:raw)` and the `ExactCholesky` solver; composes
with `match_id` (rows of one match must share a group id) and `X`;
mutually exclusive with `error_cov`.
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
    X::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    error_cov::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    error_groups::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    error_group_cap::Integer=8,
    error_blocks::Union{Nothing,Symbol}=nothing,
    firm_group::Union{Nothing,AbstractVector{<:Integer}}=nothing,
) where {M<:AbstractBipartiteModel}
    model_adjacency in (:binary, :counts) ||
        throw(ArgumentError("model_adjacency must be :binary or :counts; got $(model_adjacency)."))
    M <: BipartiteVarianceStableModel && weighting.observations != :raw &&
        throw(ArgumentError("BipartiteVarianceStableModel currently supports only Weighting(observations=:raw)."))
    match_id !== nothing && weighting.observations != :raw &&
        throw(ArgumentError("match_id grouping currently supports only Weighting(observations=:raw)."))
    if error_cov !== nothing
        weighting.observations == :raw ||
            throw(ArgumentError("error_cov currently supports only Weighting(observations=:raw)."))
        size(error_cov) == (length(y), length(y)) ||
            throw(ArgumentError("error_cov must be $(length(y))×$(length(y)) (one row per observation)."))
        issymmetric(error_cov) ||
            throw(ArgumentError("error_cov must be symmetric."))
        all(isfinite, nonzeros(sparse(error_cov))) ||
            throw(ArgumentError("error_cov entries must be finite."))
    end
    if error_groups !== nothing
        error_cov === nothing ||
            throw(ArgumentError("error_groups and error_cov are mutually exclusive."))
        weighting.observations == :raw ||
            throw(ArgumentError("error_groups currently supports only Weighting(observations=:raw)."))
        length(error_groups) == length(y) ||
            throw(ArgumentError("error_groups must have the same length as y."))
        error_group_cap >= 2 ||
            throw(ArgumentError("error_group_cap must be at least 2."))
    end
    if error_blocks !== nothing
        error_blocks == :free ||
            throw(ArgumentError("error_blocks must be :free; got $(error_blocks)."))
        error_cov === nothing ||
            throw(ArgumentError("error_blocks and error_cov are mutually exclusive."))
        error_groups === nothing ||
            throw(ArgumentError("error_blocks and error_groups are mutually exclusive."))
        weighting.observations == :raw ||
            throw(ArgumentError("error_blocks=:free currently supports only Weighting(observations=:raw)."))
        firm_group !== nothing ||
            throw(ArgumentError("error_blocks=:free requires firm_group (firm id per row)."))
        length(firm_group) == length(y) ||
            throw(ArgumentError("firm_group must have the same length as y."))
        match_id === nothing ||
            throw(ArgumentError("error_blocks=:free does not yet support match_id."))
    end

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

    if X !== nothing
        size(X, 1) == length(y) ||
            throw(ArgumentError("X must have $(length(y)) rows; got $(size(X, 1))."))
        size(X, 2) > 0 ||
            throw(ArgumentError("X must have at least one column."))
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
    banded_aux = nothing
    grouped_aux = nothing
    free_aux = nothing
    block = if obs == :raw
        if error_cov !== nothing
            R_obs = SparseMatrixCSC{Float64,Int}(sparse(error_cov)[obs_mask, obs_mask])
            design, logdet_R, n_banded, banded_aux =
                build_correlated_V_stats(f_obs, w_obs, y_obs_scaled, match_id_obs,
                                         R_obs, n_f, n_w)
            (
                K = n_banded,
                base = EdgeData(f_obs, w_obs, y_obs_scaled, ones(Int, personyear_rows)),
                design = design,
                # logdet R rides in log_weight_sum: 2*NLL contains
                # -log_weight_sum, and the correlated-error constant is
                # +logdet R, so store the negative. Weights are unit.
                weights = WeightStats(-logdet_R, Float64(n_banded), Float64(n_banded), 1.0, 1.0),
                within_ss = 0.0,
                within_df = 0,
                rho_eps_likelihood = nothing,
            )
        elseif error_groups !== nothing
            groups_obs = Int.(collect(error_groups))[obs_mask]
            design, n_groups, ec_core, aux =
                build_grouped_V_stats(f_obs, w_obs, y_obs_scaled, match_id_obs,
                                      groups_obs, Int(error_group_cap), n_f, n_w)
            grouped_aux = merge(ec_core, aux)
            (
                K = n_groups,
                base = EdgeData(f_obs, w_obs, y_obs_scaled, ones(Int, personyear_rows)),
                design = design,
                # The omega part of the likelihood constant is
                # parameter-dependent and assembled per evaluation in
                # objective_stats; the stored weights are the omega = 1 point.
                weights = trivial_weight_stats(n_groups),
                within_ss = 0.0,
                within_df = 0,
                rho_eps_likelihood = nothing,
            )
        elseif error_blocks !== nothing
            blocks_obs = Int.(collect(firm_group))[obs_mask]
            Vfb, yfb, block_of, sz, dsz, dof_b =
                build_freeblock_V_stats(f_obs, w_obs, y_obs_scaled, blocks_obs, n_f, n_w)
            free_aux = (V = Vfb, y = yfb, block_of = block_of, sizes = sz,
                        distinct = dsz, dof_blocks = dof_b)
            (
                K = length(yfb),
                base = EdgeData(f_obs, w_obs, y_obs_scaled, ones(Int, personyear_rows)),
                # The iid edge-level design is only a placeholder; the EM reads
                # V and y from `free_blocks` directly.
                design = build_V_stats(f_obs, w_obs, y_obs_scaled, n_f, n_w),
                weights = trivial_weight_stats(length(yfb)),
                within_ss = 0.0,
                within_df = 0,
                rho_eps_likelihood = nothing,
            )
        else
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
        end
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

    # ── Mean-structure statistics ──
    grouped_mean = nothing
    mean_stats = if X !== nothing
        X_obs = Matrix{Float64}(X[obs_mask, :])
        if banded_aux !== nothing
            build_correlated_mean_stats(banded_aux, X_obs[banded_aux.src, :])
        elseif grouped_aux !== nothing
            Xg = Matrix{Float64}(grouped_aux.G * X_obs[grouped_aux.src, :])
            p_cols = size(Xg, 2)
            grouped_mean = map(grouped_aux.class_idx) do idx
                Vc = grouped_aux.Vg[idx, :]
                Xc = Xg[idx, :]
                yc = grouped_aux.yg[idx]
                MeanStats(Matrix{Float64}(transpose(Vc) * Xc),
                          transpose(Xc) * Xc, vec(transpose(Xc) * yc), p_cols)
            end
            MeanStats(sum(ms.VtX for ms in grouped_mean),
                      sum(ms.XtX for ms in grouped_mean),
                      sum(ms.Xty for ms in grouped_mean), p_cols)
        elseif obs == :raw
            if match_id_obs !== nothing
                build_match_mean_stats(f_obs, w_obs, y_obs_scaled, X_obs, match_id_obs, n_f, n_w)
            else
                build_mean_stats(f_obs, w_obs, y_obs_scaled, X_obs, n_f, n_w)
            end
        elseif obs == :edge
            build_weighted_mean_stats(edges.f, edges.w, edges.y_mean, X_obs[1:n_edges, :],
                ones(Float64, n_edges), n_f, n_w)
        else  # :effective
            build_match_weight_mean_stats(edges.f, edges.w, edges.y_mean, X_obs[1:n_edges, :],
                edges.T, n_f, n_w, weighting.rho_eps)
        end
    else
        nothing
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
        correlated_errors = error_cov !== nothing,
        error_groups = error_groups !== nothing,
        error_blocks = error_blocks !== nothing,
    )

    error_classes = grouped_aux === nothing ? nothing :
        ErrorClassStats(grouped_aux.sizes, grouped_aux.counts, grouped_aux.vtv_nzvals,
                        grouped_aux.projected, grouped_aux.ydot, grouped_mean)

    free_blocks = free_aux === nothing ? nothing :
        FreeBlockStats(free_aux.V, free_aux.y, free_aux.block_of, free_aux.sizes,
                       free_aux.distinct, free_aux.dof_blocks)

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
        mean_stats,
        error_classes,
        free_blocks,
        metadata,
    )
end
